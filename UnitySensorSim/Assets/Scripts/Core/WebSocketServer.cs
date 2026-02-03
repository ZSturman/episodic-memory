using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Lightweight WebSocket server for Unity.
    /// Handles both sensor streaming (server → client) and command receiving (client → server).
    /// 
    /// Uses a simple WebSocket implementation to avoid external dependencies.
    /// </summary>
    public class WebSocketServer : MonoBehaviour
    {
        [Header("Server Configuration")]
        [SerializeField] private int port = 8765;
        [SerializeField] private bool autoStart = true;

        private TcpListener _listener;
        private List<ClientConnection> _clients = new List<ClientConnection>();
        private Thread _acceptThread;
        private volatile bool _isRunning;
        private readonly object _clientLock = new object();

        // Events
        public event Action<string> OnMessageReceived;
#pragma warning disable CS0067
        public event Action<int> OnClientConnected;
        public event Action<int> OnClientDisconnected;
#pragma warning restore CS0067

        // Public properties
        public bool IsRunning => _isRunning;
        public int ConnectedClients
        {
            get
            {
                lock (_clientLock)
                {
                    return _clients.Count;
                }
            }
        }
        public int Port => port;

        private void Start()
        {
            if (autoStart)
            {
                StartServer();
            }
        }

        private void OnDestroy()
        {
            StopServer();
        }

        private void Update()
        {
            // Process incoming messages on main thread
            ProcessIncomingMessages();
            
            // Clean up disconnected clients
            CleanupDisconnectedClients();
        }

        public void StartServer()
        {
            if (_isRunning) return;

            try
            {
                _listener = new TcpListener(IPAddress.Any, port);
                _listener.Start();
                _isRunning = true;

                _acceptThread = new Thread(AcceptClients);
                _acceptThread.IsBackground = true;
                _acceptThread.Start();

                Debug.Log($"[WebSocketServer] Started on port {port}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[WebSocketServer] Failed to start: {ex.Message}");
            }
        }

        public void StopServer()
        {
            _isRunning = false;

            lock (_clientLock)
            {
                foreach (var client in _clients)
                {
                    client.Close();
                }
                _clients.Clear();
            }

            _listener?.Stop();
            _acceptThread?.Join(1000);

            Debug.Log("[WebSocketServer] Stopped");
        }

        private void AcceptClients()
        {
            while (_isRunning)
            {
                try
                {
                    if (_listener.Pending())
                    {
                        var tcpClient = _listener.AcceptTcpClient();
                        var client = new ClientConnection(tcpClient);
                        
                        if (client.PerformHandshake())
                        {
                            lock (_clientLock)
                            {
                                _clients.Add(client);
                            }
                            client.StartReceiving();
                            Debug.Log($"[WebSocketServer] Client connected (total: {ConnectedClients})");
                        }
                        else
                        {
                            client.Close();
                        }
                    }
                    else
                    {
                        Thread.Sleep(10);
                    }
                }
                catch (Exception ex)
                {
                    if (_isRunning)
                    {
                        Debug.LogError($"[WebSocketServer] Accept error: {ex.Message}");
                    }
                }
            }
        }

        private void ProcessIncomingMessages()
        {
            List<string> messages = new List<string>();

            lock (_clientLock)
            {
                foreach (var client in _clients)
                {
                    string msg;
                    while (client.TryDequeueMessage(out msg))
                    {
                        messages.Add(msg);
                    }
                }
            }

            foreach (var msg in messages)
            {
                OnMessageReceived?.Invoke(msg);
            }
        }

        private void CleanupDisconnectedClients()
        {
            lock (_clientLock)
            {
                for (int i = _clients.Count - 1; i >= 0; i--)
                {
                    if (!_clients[i].IsConnected)
                    {
                        _clients[i].Close();
                        _clients.RemoveAt(i);
                        Debug.Log($"[WebSocketServer] Client disconnected (total: {_clients.Count})");
                    }
                }
            }
        }

        /// <summary>
        /// Broadcast a message to all connected clients.
        /// </summary>
        public void Broadcast(string message)
        {
            byte[] frame = CreateWebSocketFrame(message);

            lock (_clientLock)
            {
                foreach (var client in _clients)
                {
                    client.Send(frame);
                }
            }
        }

        /// <summary>
        /// Send a message to a specific client.
        /// </summary>
        public void SendToClient(int clientIndex, string message)
        {
            byte[] frame = CreateWebSocketFrame(message);

            lock (_clientLock)
            {
                if (clientIndex >= 0 && clientIndex < _clients.Count)
                {
                    _clients[clientIndex].Send(frame);
                }
            }
        }

        private byte[] CreateWebSocketFrame(string message)
        {
            byte[] payload = Encoding.UTF8.GetBytes(message);
            int payloadLength = payload.Length;

            byte[] frame;
            int headerLength;

            if (payloadLength <= 125)
            {
                headerLength = 2;
                frame = new byte[headerLength + payloadLength];
                frame[1] = (byte)payloadLength;
            }
            else if (payloadLength <= 65535)
            {
                headerLength = 4;
                frame = new byte[headerLength + payloadLength];
                frame[1] = 126;
                frame[2] = (byte)(payloadLength >> 8);
                frame[3] = (byte)(payloadLength & 0xFF);
            }
            else
            {
                headerLength = 10;
                frame = new byte[headerLength + payloadLength];
                frame[1] = 127;
                for (int i = 0; i < 8; i++)
                {
                    frame[9 - i] = (byte)(payloadLength >> (8 * i));
                }
            }

            // FIN bit + text opcode
            frame[0] = 0x81;
            Array.Copy(payload, 0, frame, headerLength, payloadLength);

            return frame;
        }

        /// <summary>
        /// Individual client connection handler.
        /// </summary>
        private class ClientConnection
        {
            private TcpClient _tcpClient;
            private NetworkStream _stream;
            private Thread _receiveThread;
            private Queue<string> _messageQueue = new Queue<string>();
            private readonly object _queueLock = new object();
            private volatile bool _isConnected;

            public bool IsConnected => _isConnected && _tcpClient?.Connected == true;

            public ClientConnection(TcpClient tcpClient)
            {
                _tcpClient = tcpClient;
                _stream = tcpClient.GetStream();
            }

            public bool PerformHandshake()
            {
                try
                {
                    byte[] buffer = new byte[4096];
                    int bytesRead = _stream.Read(buffer, 0, buffer.Length);
                    string request = Encoding.UTF8.GetString(buffer, 0, bytesRead);

                    if (!request.Contains("Upgrade: websocket"))
                    {
                        return false;
                    }

                    // Extract Sec-WebSocket-Key
                    string key = null;
                    foreach (string line in request.Split('\n'))
                    {
                        if (line.StartsWith("Sec-WebSocket-Key:"))
                        {
                            key = line.Substring(18).Trim();
                            break;
                        }
                    }

                    if (key == null) return false;

                    // Generate accept key
                    string acceptKey = ComputeAcceptKey(key);

                    // Send handshake response
                    string response =
                        "HTTP/1.1 101 Switching Protocols\r\n" +
                        "Upgrade: websocket\r\n" +
                        "Connection: Upgrade\r\n" +
                        $"Sec-WebSocket-Accept: {acceptKey}\r\n\r\n";

                    byte[] responseBytes = Encoding.UTF8.GetBytes(response);
                    _stream.Write(responseBytes, 0, responseBytes.Length);

                    _isConnected = true;
                    return true;
                }
                catch
                {
                    return false;
                }
            }

            private string ComputeAcceptKey(string key)
            {
                string combined = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
                byte[] hash = System.Security.Cryptography.SHA1.Create()
                    .ComputeHash(Encoding.UTF8.GetBytes(combined));
                return Convert.ToBase64String(hash);
            }

            public void StartReceiving()
            {
                _receiveThread = new Thread(ReceiveLoop);
                _receiveThread.IsBackground = true;
                _receiveThread.Start();
            }

            private void ReceiveLoop()
            {
                byte[] buffer = new byte[65536];

                while (_isConnected)
                {
                    try
                    {
                        if (_stream.DataAvailable)
                        {
                            int bytesRead = _stream.Read(buffer, 0, buffer.Length);
                            if (bytesRead == 0)
                            {
                                _isConnected = false;
                                break;
                            }

                            string message = DecodeWebSocketFrame(buffer, bytesRead);
                            if (message != null)
                            {
                                lock (_queueLock)
                                {
                                    _messageQueue.Enqueue(message);
                                }
                            }
                        }
                        else
                        {
                            Thread.Sleep(1);
                        }
                    }
                    catch
                    {
                        _isConnected = false;
                        break;
                    }
                }
            }

            private string DecodeWebSocketFrame(byte[] buffer, int length)
            {
                if (length < 2) return null;

                byte opcode = (byte)(buffer[0] & 0x0F);
                
                // Check for close frame
                if (opcode == 0x08)
                {
                    _isConnected = false;
                    return null;
                }

                // Only handle text frames
                if (opcode != 0x01) return null;

                bool masked = (buffer[1] & 0x80) != 0;
                int payloadLength = buffer[1] & 0x7F;
                int headerLength = 2;

                if (payloadLength == 126)
                {
                    payloadLength = (buffer[2] << 8) | buffer[3];
                    headerLength = 4;
                }
                else if (payloadLength == 127)
                {
                    // For simplicity, only handle up to int.MaxValue
                    payloadLength = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        payloadLength = (payloadLength << 8) | buffer[2 + i];
                    }
                    headerLength = 10;
                }

                if (masked)
                {
                    byte[] mask = new byte[4];
                    Array.Copy(buffer, headerLength, mask, 0, 4);
                    headerLength += 4;

                    byte[] payload = new byte[payloadLength];
                    for (int i = 0; i < payloadLength; i++)
                    {
                        payload[i] = (byte)(buffer[headerLength + i] ^ mask[i % 4]);
                    }

                    return Encoding.UTF8.GetString(payload);
                }
                else
                {
                    return Encoding.UTF8.GetString(buffer, headerLength, payloadLength);
                }
            }

            public bool TryDequeueMessage(out string message)
            {
                lock (_queueLock)
                {
                    if (_messageQueue.Count > 0)
                    {
                        message = _messageQueue.Dequeue();
                        return true;
                    }
                }
                message = null;
                return false;
            }

            public void Send(byte[] data)
            {
                if (!_isConnected) return;

                try
                {
                    _stream.Write(data, 0, data.Length);
                }
                catch
                {
                    _isConnected = false;
                }
            }

            public void Close()
            {
                _isConnected = false;
                _stream?.Close();
                _tcpClient?.Close();
            }
        }
    }
}

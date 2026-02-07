using System;
using UnityEngine;
using EpisodicAgent.Protocol;
using EpisodicAgent.World;
using EpisodicAgent.Player;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Streams sensor frames to connected Python clients at a configurable rate.
    /// Sends camera pose (world position + orientation) and current room occupancy.
    /// ARCHITECTURAL INVARIANT: Unity = eyes, backend = brain.
    /// Entity discovery happens through the Visual Summary Channel (port 8767),
    /// NOT through pre-identified object lists.
    /// </summary>
    public class SensorStreamer : MonoBehaviour
    {
        [Header("Streaming Configuration")]
        [SerializeField] private float targetFrameRate = 10f;  // Frames per second
        [SerializeField] private bool streamWhenConnected = true;
        [Header("References")]
        [SerializeField] private WebSocketServer webSocketServer;
        [SerializeField] private Transform playerCamera;
        [SerializeField] private WorldManager worldManager;

        [Header("Debug")]
        [SerializeField] private bool debugLogging = false;

        // State tracking
        private int _frameId = 0;
        private float _lastSendTime;
        private float _sendInterval;

        // Public properties
        public int LastFrameId => _frameId;
        public float ActualFrameRate { get; private set; }
        public bool IsStreaming => webSocketServer != null && webSocketServer.ConnectedClients > 0;

        // Events
        public event Action<int> OnFrameSent;
        /// <summary>Fires with the raw JSON string right before it is broadcast. Useful for debugging.</summary>
        public event Action<string> OnFrameJsonSent;

        private void Start()
        {
            _sendInterval = 1f / targetFrameRate;
            _lastSendTime = Time.time;

            if (webSocketServer == null)
            {
                webSocketServer = GetComponent<WebSocketServer>();
            }

            if (worldManager == null)
            {
                worldManager = FindFirstObjectByType<WorldManager>();
            }

            // Auto-discover playerCamera if not wired in Inspector
            if (playerCamera == null)
            {
                Camera mainCam = Camera.main;
                if (mainCam != null)
                {
                    playerCamera = mainCam.transform;
                    Debug.Log($"[SensorStreamer] Auto-discovered playerCamera from Camera.main: {playerCamera.name}");
                }
            }

            if (playerCamera == null)
            {
                // Try to find a FirstPersonController's camera as fallback
                var fpc = FindFirstObjectByType<EpisodicAgent.Player.FirstPersonController>();
                if (fpc != null)
                {
                    Camera cam = fpc.GetComponentInChildren<Camera>();
                    if (cam != null)
                    {
                        playerCamera = cam.transform;
                        Debug.Log($"[SensorStreamer] Auto-discovered playerCamera from FirstPersonController: {playerCamera.name}");
                    }
                }
            }

            if (playerCamera == null)
            {
                Debug.LogWarning("[SensorStreamer] playerCamera is NULL — all camera data will be default. " +
                    "Assign the player camera in the Inspector or ensure Camera.main exists.");
            }
        }

        private void OnDestroy()
        {
            // No subscriptions to clean up — entity tracking removed
        }

        private void Update()
        {
            if (!streamWhenConnected || webSocketServer == null)
                return;

            if (webSocketServer.ConnectedClients == 0)
                return;

            float elapsed = Time.time - _lastSendTime;
            if (elapsed >= _sendInterval)
            {
                SendSensorFrame();
                ActualFrameRate = 1f / elapsed;
                _lastSendTime = Time.time;
            }
        }

        /// <summary>
        /// Manually trigger sending a sensor frame.
        /// </summary>
        public void SendSensorFrame()
        {
            if (webSocketServer == null || webSocketServer.ConnectedClients == 0)
                return;

            SensorFrame frame = BuildSensorFrame();
            string json = JsonUtility.ToJson(frame);

            // Wire-level debug: log what is actually being serialized and sent
            if (debugLogging)
            {
                var cam = frame.camera;
                Debug.Log($"[SensorStreamer] SEND frame_id={frame.frame_id} " +
                    $"pos=({cam.position.x:F2},{cam.position.y:F2},{cam.position.z:F2}) " +
                    $"fwd=({cam.forward.x:F2},{cam.forward.y:F2},{cam.forward.z:F2}) " +
                    $"yaw={cam.yaw:F1} pitch={cam.pitch:F1} " +
                    $"room={frame.current_room_guid} " +
                    $"bytes={json.Length}");
            }

            OnFrameJsonSent?.Invoke(json);

            webSocketServer.Broadcast(json);

            _frameId++;

            OnFrameSent?.Invoke(_frameId);
        }

        private SensorFrame BuildSensorFrame()
        {
            var frame = new SensorFrame
            {
                protocol_version = ProtocolVersion.VERSION,
                timestamp = DateTime.UtcNow.ToString("o"),
                frame_id = _frameId,
                camera = BuildCameraPose(),
                current_room_guid = GetCurrentRoomGuid(),
                // ARCHITECTURAL INVARIANT: Unity = eyes, backend = brain.
                // Entities are NOT sent from Unity. The backend discovers objects
                // through the Visual Summary Channel (port 8767) and learns labels
                // from user interaction. Only camera pose + room occupancy are sent.
            };

            return frame;
        }

        private CameraPose BuildCameraPose()
        {
            if (playerCamera == null)
            {
                return new CameraPose
                {
                    position = new Vector3Data(Vector3.zero),
                    forward = new Vector3Data(Vector3.forward),
                    up = new Vector3Data(Vector3.up),
                    yaw = 0,
                    pitch = 0
                };
            }

            Vector3 euler = playerCamera.eulerAngles;
            
            return new CameraPose
            {
                position = new Vector3Data(playerCamera.position),
                forward = new Vector3Data(playerCamera.forward),
                up = new Vector3Data(playerCamera.up),
                yaw = euler.y,
                pitch = euler.x > 180 ? euler.x - 360 : euler.x  // Normalize to -180..180
            };
        }

        private string GetCurrentRoomGuid()
        {
            if (worldManager == null) return "";
            var room = worldManager.CurrentRoom;
            return room != null ? room.Guid : "";
        }

        // REMOVED: Entity methods (BuildEntityList, GetEntitySize, IsEntityVisible,
        // GetInteractableType, GetInteractableState, HandleEntityStateChanged).
        // ARCHITECTURAL INVARIANT: Unity = eyes, backend = brain.
        // The backend discovers entities through the Visual Summary Channel (port 8767).

        /// <summary>
        /// Get the target streaming frame rate.
        /// </summary>
        public float TargetHz => targetFrameRate;

        /// <summary>
        /// Set the target streaming frame rate.
        /// </summary>
        public void SetTargetFrameRate(float fps)
        {
            targetFrameRate = Mathf.Clamp(fps, 1f, 60f);
            _sendInterval = 1f / targetFrameRate;
        }
    }
}

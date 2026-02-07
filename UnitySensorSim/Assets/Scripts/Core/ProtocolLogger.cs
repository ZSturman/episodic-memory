using System;
using System.IO;
using UnityEngine;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Logs all WebSocket protocol traffic (sensor frames out, commands in,
    /// responses out) to the Unity Console and optionally to a log file.
    ///
    /// Attach to the same GameObject as WebSocketServer / SensorStreamer /
    /// CommandReceiver, or assign the references in the Inspector.
    ///
    /// Toggle at runtime:
    ///   • F2 — cycle through Off → Headers → Full JSON
    ///   • F3 — toggle file logging on/off
    ///
    /// ARCHITECTURAL NOTE: This component lives entirely inside the Unity
    /// simulator. The Python backend knows nothing about it.
    /// </summary>
    public class ProtocolLogger : MonoBehaviour
    {
        // ── Inspector ──────────────────────────────────────────────────
        [Header("References (auto-detected if empty)")]
        [SerializeField] private SensorStreamer sensorStreamer;
        [SerializeField] private CommandReceiver commandReceiver;

        [Header("Console Logging")]
        [Tooltip("Off = silent, Headers = direction + size, Full = pretty-printed JSON")]
        [SerializeField] private LogVerbosity verbosity = LogVerbosity.Headers;
        [SerializeField] private KeyCode toggleVerbosityKey = KeyCode.F2;

        [Header("File Logging")]
        [SerializeField] private bool logToFile = false;
        [SerializeField] private KeyCode toggleFileKey = KeyCode.F3;
        [Tooltip("Relative to Application.persistentDataPath if not absolute")]
        [SerializeField] private string logFileName = "protocol_traffic.jsonl";

        private StreamWriter _fileWriter;
        private string _logFilePath;

        // ── Types ──────────────────────────────────────────────────────
        public enum LogVerbosity { Off, Headers, Full }

        private enum Direction { Out, In }

        // ── Public accessors (for HUD) ─────────────────────────────────
        public LogVerbosity CurrentVerbosity => verbosity;
        public bool IsFileLogging => logToFile && _fileWriter != null;
        public string LogFilePath => _logFilePath;

        // Event other components (e.g. ProtocolLogHUD) can subscribe to.
        // Args: direction ("OUT"/"IN"), category, raw json
        public event Action<string, string, string> OnTrafficLogged;

        // ── Lifecycle ──────────────────────────────────────────────────
        private void Start()
        {
            if (sensorStreamer == null) sensorStreamer = FindFirstObjectByType<SensorStreamer>();
            if (commandReceiver == null) commandReceiver = FindFirstObjectByType<CommandReceiver>();

            if (sensorStreamer != null)
                sensorStreamer.OnFrameJsonSent += HandleFrameSent;
            else
                Debug.LogWarning("[ProtocolLogger] SensorStreamer not found — frame logging disabled");

            if (commandReceiver != null)
            {
                commandReceiver.OnCommandJsonReceived += HandleCommandReceived;
                commandReceiver.OnResponseJsonSent += HandleResponseSent;
            }
            else
            {
                Debug.LogWarning("[ProtocolLogger] CommandReceiver not found — command logging disabled");
            }

            // Resolve log file path
            _logFilePath = Path.IsPathRooted(logFileName)
                ? logFileName
                : Path.Combine(Application.persistentDataPath, logFileName);

            if (logToFile)
                OpenLogFile();

            Debug.Log($"[ProtocolLogger] Started — verbosity={verbosity}, fileLog={logToFile}");
        }

        private void OnDestroy()
        {
            if (sensorStreamer != null)
                sensorStreamer.OnFrameJsonSent -= HandleFrameSent;

            if (commandReceiver != null)
            {
                commandReceiver.OnCommandJsonReceived -= HandleCommandReceived;
                commandReceiver.OnResponseJsonSent -= HandleResponseSent;
            }

            CloseLogFile();
        }

        private void Update()
        {
            if (Input.GetKeyDown(toggleVerbosityKey))
            {
                verbosity = (LogVerbosity)(((int)verbosity + 1) % 3);
                Debug.Log($"[ProtocolLogger] Verbosity → {verbosity}");
            }

            if (Input.GetKeyDown(toggleFileKey))
            {
                logToFile = !logToFile;
                if (logToFile) OpenLogFile();
                else CloseLogFile();
                Debug.Log($"[ProtocolLogger] File logging → {(logToFile ? "ON" : "OFF")} ({_logFilePath})");
            }
        }

        // ── Event handlers ─────────────────────────────────────────────

        private void HandleFrameSent(string json)
        {
            Log(Direction.Out, "SENSOR_FRAME", json);
        }

        private void HandleCommandReceived(string json)
        {
            Log(Direction.In, "COMMAND", json);
        }

        private void HandleResponseSent(string json)
        {
            Log(Direction.Out, "CMD_RESPONSE", json);
        }

        // ── Core logging ───────────────────────────────────────────────

        private void Log(Direction dir, string category, string json)
        {
            string arrow = dir == Direction.Out ? ">>>" : "<<<";
            string dirLabel = dir == Direction.Out ? "OUT" : "IN";

            // Notify subscribers (e.g. HUD)
            OnTrafficLogged?.Invoke(dirLabel, category, json);

            // Console
            if (verbosity == LogVerbosity.Headers)
            {
                Debug.Log($"[Protocol {arrow}] {category}  ({json.Length} bytes)");
            }
            else if (verbosity == LogVerbosity.Full)
            {
                // Pretty-print for readability (indent 2 spaces)
                string pretty = PrettyJson(json);
                Debug.Log($"[Protocol {arrow}] {category}\n{pretty}");
            }
            // Off → nothing to console

            // File
            if (logToFile && _fileWriter != null)
            {
                try
                {
                    // JSONL: one JSON object per line with metadata wrapper
                    string ts = DateTime.UtcNow.ToString("o");
                    _fileWriter.WriteLine(
                        $"{{\"ts\":\"{ts}\",\"dir\":\"{dirLabel}\",\"cat\":\"{category}\",\"msg\":{json}}}");
                    _fileWriter.Flush();
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[ProtocolLogger] File write error: {ex.Message}");
                }
            }
        }

        // ── File helpers ───────────────────────────────────────────────

        private void OpenLogFile()
        {
            try
            {
                string dir = Path.GetDirectoryName(_logFilePath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                _fileWriter = new StreamWriter(_logFilePath, append: true);
                Debug.Log($"[ProtocolLogger] Logging to {_logFilePath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ProtocolLogger] Cannot open log file: {ex.Message}");
                logToFile = false;
            }
        }

        private void CloseLogFile()
        {
            if (_fileWriter != null)
            {
                _fileWriter.Flush();
                _fileWriter.Close();
                _fileWriter = null;
            }
        }

        // ── Utility ────────────────────────────────────────────────────

        /// <summary>
        /// Minimal pretty-printer: adds newlines + indentation to JSON.
        /// Does not depend on any third-party library.
        /// </summary>
        private static string PrettyJson(string raw)
        {
            try
            {
                var sb = new System.Text.StringBuilder();
                int indent = 0;
                bool inString = false;
                bool escaped = false;

                foreach (char c in raw)
                {
                    if (escaped) { sb.Append(c); escaped = false; continue; }
                    if (c == '\\' && inString) { sb.Append(c); escaped = true; continue; }
                    if (c == '"') { inString = !inString; sb.Append(c); continue; }

                    if (!inString)
                    {
                        switch (c)
                        {
                            case '{':
                            case '[':
                                sb.Append(c);
                                sb.AppendLine();
                                indent++;
                                sb.Append(new string(' ', indent * 2));
                                break;
                            case '}':
                            case ']':
                                sb.AppendLine();
                                indent--;
                                sb.Append(new string(' ', indent * 2));
                                sb.Append(c);
                                break;
                            case ',':
                                sb.Append(c);
                                sb.AppendLine();
                                sb.Append(new string(' ', indent * 2));
                                break;
                            case ':':
                                sb.Append(": ");
                                break;
                            default:
                                if (!char.IsWhiteSpace(c))
                                    sb.Append(c);
                                break;
                        }
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }

                return sb.ToString();
            }
            catch
            {
                return raw;
            }
        }
    }
}

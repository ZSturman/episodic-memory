using System.Collections.Generic;
using UnityEngine;

namespace EpisodicAgent.UI
{
    /// <summary>
    /// In-game overlay that shows a scrollable log of recent protocol traffic
    /// (sensor frames, commands, responses) with direction arrows.
    ///
    /// Toggle visibility with F4 (configurable).
    /// Requires <see cref="EpisodicAgent.Core.ProtocolLogger"/> on the same
    /// or another GameObject.
    ///
    /// ARCHITECTURAL NOTE: Pure Unity-side debugging aid.
    /// The Python backend is unaware of this component.
    /// </summary>
    public class ProtocolLogHUD : MonoBehaviour
    {
        // ── Inspector ──────────────────────────────────────────────────
        [Header("References (auto-detected if empty)")]
        [SerializeField] private Core.ProtocolLogger protocolLogger;

        [Header("Display")]
        [SerializeField] private bool showPanel = false;
        [SerializeField] private KeyCode toggleKey = KeyCode.F4;
        [SerializeField] private int maxEntries = 40;
        [SerializeField] private int panelWidth = 700;
        [SerializeField] private int panelHeight = 500;
        [SerializeField] private int fontSize = 12;

        [Header("Filters")]
        [SerializeField] private bool showSensorFrames = true;
        [SerializeField] private bool showCommands = true;
        [SerializeField] private bool showResponses = true;

        // ── Internal state ─────────────────────────────────────────────
        private readonly List<LogEntry> _entries = new List<LogEntry>();
        private Vector2 _scrollPos;
        private bool _autoScroll = true;

        // Styles (lazy-init)
        private GUIStyle _boxStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _headerStyle;
        private GUIStyle _outStyle;
        private GUIStyle _inStyle;
        private GUIStyle _btnStyle;
        private bool _stylesInitialised;

        // ── Data ───────────────────────────────────────────────────────
        private struct LogEntry
        {
            public string Time;
            public string Direction;  // "OUT" or "IN"
            public string Category;   // "SENSOR_FRAME", "COMMAND", "CMD_RESPONSE"
            public string Json;
            public bool Expanded;
        }

        // ── Lifecycle ──────────────────────────────────────────────────
        private void Start()
        {
            if (protocolLogger == null)
                protocolLogger = FindFirstObjectByType<Core.ProtocolLogger>();

            if (protocolLogger != null)
                protocolLogger.OnTrafficLogged += HandleTraffic;
            else
                Debug.LogWarning("[ProtocolLogHUD] ProtocolLogger not found — HUD will be empty");
        }

        private void OnDestroy()
        {
            if (protocolLogger != null)
                protocolLogger.OnTrafficLogged -= HandleTraffic;
        }

        private void Update()
        {
            if (Input.GetKeyDown(toggleKey))
            {
                showPanel = !showPanel;
            }
        }

        // ── Event handler ──────────────────────────────────────────────
        private void HandleTraffic(string dir, string category, string json)
        {
            // Filter
            if (category == "SENSOR_FRAME" && !showSensorFrames) return;
            if (category == "COMMAND" && !showCommands) return;
            if (category == "CMD_RESPONSE" && !showResponses) return;

            _entries.Add(new LogEntry
            {
                Time = System.DateTime.Now.ToString("HH:mm:ss.fff"),
                Direction = dir,
                Category = category,
                Json = json,
                Expanded = false
            });

            // Trim
            while (_entries.Count > maxEntries)
                _entries.RemoveAt(0);

            // Auto-scroll
            if (_autoScroll)
                _scrollPos.y = float.MaxValue;
        }

        // ── GUI ────────────────────────────────────────────────────────
        private void OnGUI()
        {
            if (!showPanel) return;

            InitStyles();

            // Panel position: right side of screen
            float x = Screen.width - panelWidth - 10;
            float y = 10;
            Rect panelRect = new Rect(x, y, panelWidth, panelHeight);

            GUI.Box(panelRect, "", _boxStyle);

            GUILayout.BeginArea(panelRect);

            // ── Header bar ──
            GUILayout.BeginHorizontal();
            GUILayout.Label("PROTOCOL TRAFFIC", _headerStyle, GUILayout.Height(22));
            GUILayout.FlexibleSpace();

            // Filter toggles
            Color origBg = GUI.backgroundColor;

            GUI.backgroundColor = showSensorFrames ? Color.green : Color.gray;
            if (GUILayout.Button("Frames", _btnStyle, GUILayout.Width(55)))
                showSensorFrames = !showSensorFrames;

            GUI.backgroundColor = showCommands ? Color.cyan : Color.gray;
            if (GUILayout.Button("Cmds", _btnStyle, GUILayout.Width(45)))
                showCommands = !showCommands;

            GUI.backgroundColor = showResponses ? Color.yellow : Color.gray;
            if (GUILayout.Button("Resp", _btnStyle, GUILayout.Width(45)))
                showResponses = !showResponses;

            GUI.backgroundColor = origBg;

            if (GUILayout.Button("Clear", _btnStyle, GUILayout.Width(45)))
                _entries.Clear();

            string scrollLabel = _autoScroll ? "⇓" : "⇑";
            if (GUILayout.Button(scrollLabel, _btnStyle, GUILayout.Width(22)))
                _autoScroll = !_autoScroll;

            GUILayout.EndHorizontal();

            // ── Status ──
            string status = protocolLogger != null
                ? $"Verbosity: {protocolLogger.CurrentVerbosity}" +
                  (protocolLogger.IsFileLogging ? $"  |  File: {System.IO.Path.GetFileName(protocolLogger.LogFilePath)}" : "")
                : "Logger not found";
            GUILayout.Label(status, _labelStyle);

            // ── Scrollable log ──
            _scrollPos = GUILayout.BeginScrollView(_scrollPos);

            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                GUIStyle style = entry.Direction == "OUT" ? _outStyle : _inStyle;
                string arrow = entry.Direction == "OUT" ? ">>>" : "<<<";
                string header = $"[{entry.Time}] {arrow} {entry.Category} ({entry.Json.Length}B)";

                GUILayout.BeginHorizontal();

                if (GUILayout.Button(entry.Expanded ? "▼" : "►", _btnStyle, GUILayout.Width(20)))
                {
                    entry.Expanded = !entry.Expanded;
                    _entries[i] = entry;
                }

                GUILayout.Label(header, style);
                GUILayout.EndHorizontal();

                if (entry.Expanded)
                {
                    // Truncate very large payloads for display
                    string display = entry.Json.Length > 4000
                        ? entry.Json.Substring(0, 4000) + "\n... (truncated)"
                        : entry.Json;

                    GUILayout.Label(display, _labelStyle);
                }
            }

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        // ── Style init ─────────────────────────────────────────────────
        private void InitStyles()
        {
            if (_stylesInitialised) return;
            _stylesInitialised = true;

            _boxStyle = new GUIStyle(GUI.skin.box);
            _boxStyle.normal.background = MakeTex(2, 2, new Color(0, 0, 0, 0.85f));

            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = fontSize,
                wordWrap = true,
                richText = true
            };
            _labelStyle.normal.textColor = Color.white;

            _headerStyle = new GUIStyle(_labelStyle)
            {
                fontStyle = FontStyle.Bold,
                fontSize = fontSize + 2
            };
            _headerStyle.normal.textColor = new Color(0.4f, 0.8f, 1f);

            _outStyle = new GUIStyle(_labelStyle);
            _outStyle.normal.textColor = new Color(0.5f, 1f, 0.5f); // green for outgoing

            _inStyle = new GUIStyle(_labelStyle);
            _inStyle.normal.textColor = new Color(1f, 0.8f, 0.4f); // orange for incoming

            _btnStyle = new GUIStyle(GUI.skin.button) { fontSize = fontSize - 1 };
        }

        private Texture2D MakeTex(int w, int h, Color col)
        {
            Color[] px = new Color[w * h];
            for (int i = 0; i < px.Length; i++) px[i] = col;
            var tex = new Texture2D(w, h);
            tex.SetPixels(px);
            tex.Apply();
            return tex;
        }
    }
}

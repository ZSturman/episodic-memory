using UnityEngine;
using EpisodicAgent.Core;
using EpisodicAgent.World;
using EpisodicAgent.Player;

namespace EpisodicAgent.UI
{
    /// <summary>
    /// HUD displaying connection status, frame rate, and sensor streaming info.
    /// </summary>
    public class ConnectionHUD : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private WebSocketServer webSocketServer;
        [SerializeField] private SensorStreamer sensorStreamer;
        [SerializeField] private WorldManager worldManager;
        [SerializeField] private PlayerInteraction playerInteraction;

        [Header("Display Settings")]
        [SerializeField] private bool showHUD = true;
        [SerializeField] private KeyCode toggleKey = KeyCode.F1;

        [Header("Layout")]
        [SerializeField] private int padding = 10;
        [SerializeField] private int lineHeight = 20;
        [SerializeField] private int fontSize = 14;

        // Cached data
        private GUIStyle boxStyle;
        private GUIStyle labelStyle;
        private GUIStyle headerStyle;
        private float fps;
        private float fpsUpdateTimer;
        private int frameCount;

        private void Awake()
        {
            // Auto-find components
            if (webSocketServer == null)
            {
                webSocketServer = FindObjectOfType<WebSocketServer>();
            }
            if (sensorStreamer == null)
            {
                sensorStreamer = FindObjectOfType<SensorStreamer>();
            }
            if (worldManager == null)
            {
                worldManager = FindObjectOfType<WorldManager>();
            }
            if (playerInteraction == null)
            {
                playerInteraction = FindObjectOfType<PlayerInteraction>();
            }
        }

        private void Update()
        {
            // Toggle HUD
            if (Input.GetKeyDown(toggleKey))
            {
                showHUD = !showHUD;
            }

            // Calculate FPS
            frameCount++;
            fpsUpdateTimer += Time.unscaledDeltaTime;
            if (fpsUpdateTimer >= 0.5f)
            {
                fps = frameCount / fpsUpdateTimer;
                frameCount = 0;
                fpsUpdateTimer = 0;
            }
        }

        private void OnGUI()
        {
            if (!showHUD) return;

            InitStyles();

            // Connection status panel (top-left)
            DrawConnectionPanel();

            // Interaction prompt (bottom-center)
            DrawInteractionPrompt();
        }

        private void InitStyles()
        {
            if (boxStyle == null)
            {
                boxStyle = new GUIStyle(GUI.skin.box);
                boxStyle.normal.background = MakeTexture(2, 2, new Color(0, 0, 0, 0.7f));
            }

            if (labelStyle == null)
            {
                labelStyle = new GUIStyle(GUI.skin.label);
                labelStyle.fontSize = fontSize;
                labelStyle.normal.textColor = Color.white;
            }

            if (headerStyle == null)
            {
                headerStyle = new GUIStyle(labelStyle);
                headerStyle.fontStyle = FontStyle.Bold;
                headerStyle.normal.textColor = new Color(0.4f, 0.8f, 1f);
            }
        }

        private void DrawConnectionPanel()
        {
            int panelWidth = 250;
            int lines = 8;
            int panelHeight = lines * lineHeight + padding * 2;

            Rect panelRect = new Rect(padding, padding, panelWidth, panelHeight);
            GUI.Box(panelRect, "", boxStyle);

            int y = padding + 5;
            int x = padding + 10;

            // Header
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), "SENSOR SIMULATOR", headerStyle);
            y += lineHeight;

            // Connection status
            bool isConnected = webSocketServer != null && webSocketServer.IsRunning;
            int clientCount = webSocketServer?.ClientCount ?? 0;
            
            string statusText = isConnected 
                ? $"● Connected ({clientCount} client{(clientCount != 1 ? "s" : "")})" 
                : "○ Disconnected";
            Color statusColor = isConnected ? Color.green : Color.gray;
            
            GUIStyle statusStyle = new GUIStyle(labelStyle);
            statusStyle.normal.textColor = statusColor;
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), statusText, statusStyle);
            y += lineHeight;

            // Port
            int port = webSocketServer?.Port ?? 0;
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), $"Port: {port}", labelStyle);
            y += lineHeight;

            // Frame rate
            float targetHz = sensorStreamer?.TargetFrameRate ?? 0;
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), 
                $"Sensor Rate: {targetHz:F1} Hz", labelStyle);
            y += lineHeight;

            // Last frame ID
            long frameId = sensorStreamer?.LastFrameId ?? 0;
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), 
                $"Last Frame: #{frameId}", labelStyle);
            y += lineHeight;

            // Current room
            string roomName = worldManager?.CurrentRoom?.Label ?? "None";
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), 
                $"Room: {roomName}", labelStyle);
            y += lineHeight;

            // FPS
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), 
                $"FPS: {fps:F1}", labelStyle);
            y += lineHeight;

            // Toggle hint
            GUIStyle hintStyle = new GUIStyle(labelStyle);
            hintStyle.normal.textColor = Color.gray;
            hintStyle.fontSize = fontSize - 2;
            GUI.Label(new Rect(x, y, panelWidth, lineHeight), 
                $"Press {toggleKey} to toggle HUD", hintStyle);
        }

        private void DrawInteractionPrompt()
        {
            if (playerInteraction == null || !playerInteraction.HasTarget) return;

            string prompt = playerInteraction.InteractionPrompt;
            if (string.IsNullOrEmpty(prompt)) return;

            // Calculate size
            GUIStyle promptStyle = new GUIStyle(labelStyle);
            promptStyle.alignment = TextAnchor.MiddleCenter;
            promptStyle.fontSize = fontSize + 2;
            promptStyle.normal.textColor = Color.yellow;

            Vector2 size = promptStyle.CalcSize(new GUIContent(prompt));
            float x = (Screen.width - size.x) / 2f;
            float y = Screen.height - 100;

            // Draw background
            Rect bgRect = new Rect(x - 10, y - 5, size.x + 20, size.y + 10);
            GUI.Box(bgRect, "", boxStyle);

            // Draw text
            GUI.Label(new Rect(x, y, size.x, size.y), prompt, promptStyle);
        }

        private Texture2D MakeTexture(int width, int height, Color color)
        {
            Color[] pixels = new Color[width * height];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = color;
            }

            Texture2D texture = new Texture2D(width, height);
            texture.SetPixels(pixels);
            texture.Apply();
            return texture;
        }
    }
}

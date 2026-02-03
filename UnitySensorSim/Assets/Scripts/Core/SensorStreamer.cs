using System;
using System.Collections.Generic;
using UnityEngine;
using EpisodicAgent.Protocol;
using EpisodicAgent.World;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Streams sensor frames to connected Python clients at a configurable rate.
    /// Collects data from RoomVolumes, EntityMarkers, and InteractableStates.
    /// </summary>
    public class SensorStreamer : MonoBehaviour
    {
        [Header("Streaming Configuration")]
        [SerializeField] private float targetFrameRate = 10f;  // Frames per second
        [SerializeField] private bool streamWhenConnected = true;
        [SerializeField] private bool includeAllEntities = true;  // Cheat mode: include even non-visible entities

        [Header("References")]
        [SerializeField] private WebSocketServer webSocketServer;
        [SerializeField] private Transform playerCamera;
        [SerializeField] private WorldManager worldManager;

        // State tracking
        private int _frameId = 0;
        private float _lastSendTime;
        private float _sendInterval;
        private List<StateChangeEvent> _pendingStateChanges = new List<StateChangeEvent>();
        private Dictionary<string, string> _lastEntityStates = new Dictionary<string, string>();

        // Public properties
        public int LastFrameId => _frameId;
        public float ActualFrameRate { get; private set; }
        public bool IsStreaming => webSocketServer != null && webSocketServer.ConnectedClients > 0;

        // Events
        public event Action<int> OnFrameSent;

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

            // Subscribe to interactable state changes
            if (worldManager != null)
            {
                worldManager.OnEntityStateChanged += HandleEntityStateChangedWrapper;
            }
        }

        private void OnDestroy()
        {
            if (worldManager != null)
            {
                worldManager.OnEntityStateChanged -= HandleEntityStateChangedWrapper;
            }
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
            webSocketServer.Broadcast(json);

            _frameId++;
            _pendingStateChanges.Clear();

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
                current_room_label = GetCurrentRoomLabel(),
                entities = BuildEntityList(),
                state_changes = new List<StateChangeEvent>(_pendingStateChanges)
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

        private string GetCurrentRoomLabel()
        {
            if (worldManager == null) return "unknown";
            var room = worldManager.CurrentRoom;
            return room != null ? room.Label : "unknown";
        }

        private List<EntityData> BuildEntityList()
        {
            var entities = new List<EntityData>();

            if (worldManager == null) return entities;

            foreach (var marker in worldManager.Entities)
            {
                if (marker == null) continue;

                bool isVisible = IsEntityVisible(marker);
                
                if (!includeAllEntities && !isVisible)
                    continue;

                var entityData = new EntityData
                {
                    guid = marker.Guid,
                    label = marker.Label,
                    category = marker.Category,
                    position = new Vector3Data(marker.transform.position),
                    size = new Vector3Data(GetEntitySize(marker)),
                    distance = playerCamera != null 
                        ? Vector3.Distance(playerCamera.position, marker.transform.position) 
                        : 0f,
                    is_visible = isVisible,
                    interactable_type = GetInteractableType(marker),
                    interactable_state = GetInteractableState(marker)
                };

                entities.Add(entityData);

                // Track state for change detection
                string currentState = entityData.interactable_state ?? "";
                string lastState;
                if (_lastEntityStates.TryGetValue(marker.Guid, out lastState))
                {
                    if (currentState != lastState)
                    {
                        // State changed - already captured via event
                    }
                }
                _lastEntityStates[marker.Guid] = currentState;
            }

            return entities;
        }

        private Vector3 GetEntitySize(EntityMarker marker)
        {
            // Try to get size from renderer bounds
            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                return renderer.bounds.size;
            }

            // Try to get size from collider
            Collider collider = marker.GetComponent<Collider>();
            if (collider != null)
            {
                return collider.bounds.size;
            }

            // Default size
            return Vector3.one;
        }

        private bool IsEntityVisible(EntityMarker marker)
        {
            if (playerCamera == null) return true;

            // Simple visibility check using viewport
            Camera cam = playerCamera.GetComponent<Camera>();
            if (cam == null) cam = Camera.main;
            if (cam == null) return true;

            Vector3 viewportPoint = cam.WorldToViewportPoint(marker.transform.position);
            
            // Check if in viewport
            if (viewportPoint.z < 0 || viewportPoint.x < 0 || viewportPoint.x > 1 ||
                viewportPoint.y < 0 || viewportPoint.y > 1)
            {
                return false;
            }

            // Optional: raycast check for occlusion
            Vector3 direction = marker.transform.position - playerCamera.position;
            RaycastHit hit;
            if (Physics.Raycast(playerCamera.position, direction, out hit, direction.magnitude))
            {
                // Check if we hit the entity or something closer
                if (hit.transform != marker.transform && 
                    !hit.transform.IsChildOf(marker.transform))
                {
                    return false;
                }
            }

            return true;
        }

        private string GetInteractableType(EntityMarker marker)
        {
            var interactable = marker.GetComponent<InteractableState>();
            if (interactable == null) return null;
            return interactable.Type.ToString().ToLower();
        }

        private string GetInteractableState(EntityMarker marker)
        {
            var interactable = marker.GetComponent<InteractableState>();
            if (interactable == null) return null;
            return interactable.CurrentState;
        }

        private void HandleEntityStateChangedWrapper(EntityMarker marker, string oldState, string newState)
        {
            if (marker == null) return;
            HandleEntityStateChanged(marker.Guid, oldState, newState);
        }

        private void HandleEntityStateChanged(string entityGuid, string oldState, string newState)
        {
            var change = new StateChangeEvent
            {
                entity_guid = entityGuid,
                change_type = "state_changed",
                old_value = oldState,
                new_value = newState,
                timestamp = DateTime.UtcNow.ToString("o")
            };
            _pendingStateChanges.Add(change);
        }

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

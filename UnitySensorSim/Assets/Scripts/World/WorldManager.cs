using System;
using System.Collections.Generic;
using UnityEngine;
using EpisodicAgent.Protocol;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Central manager for all rooms, entities, and world state.
    /// Provides queries for sensor streaming and command handling.
    /// </summary>
    public class WorldManager : MonoBehaviour
    {
        [Header("World Configuration")]
        [SerializeField] private Transform playerTransform;
        [SerializeField] private Camera playerCamera;

        [Header("Entity Auto-Discovery (disabled by default)")]
        [Tooltip("DISABLED: Auto-discovery violates architecture (Unity=eyes, backend=brain). Only manually placed EntityMarkers are tracked.")]
        [SerializeField] private bool autoDiscoverEntities = false;
        [Tooltip("Minimum renderer bounds volume to qualify for auto-discovery (filters tiny/invisible objects).")]
        [SerializeField] private float minEntityVolume = 0.01f;
        [Tooltip("Layers to exclude from auto-discovery (e.g. UI, Terrain, Water).")]
        [SerializeField] private LayerMask autoDiscoverExcludeLayers;

        [Header("Spawnable Prefabs")]
        [SerializeField] private GameObject ballPrefab;

        [Header("Debug")]
        [SerializeField] private bool debugLogging = false;

        // Collections
        private List<RoomVolume> rooms = new List<RoomVolume>();
        private List<EntityMarker> entities = new List<EntityMarker>();
        private List<StateChangeEvent> pendingStateChanges = new List<StateChangeEvent>();
        private long frameIdCounter = 0;

        // Spawned objects tracking
        private GameObject spawnedBall;
        private EntityMarker spawnedBallMarker;

        // Events
        public event Action<RoomVolume> OnRoomEntered;
        public event Action<RoomVolume> OnRoomExited;
        public event Action<EntityMarker, string, string> OnEntityStateChanged;

        // Public accessors
        public RoomVolume CurrentRoom { get; private set; }
        public Transform Player => playerTransform;
        public Camera PlayerCamera => playerCamera;
        public IReadOnlyList<RoomVolume> Rooms => rooms;
        public IReadOnlyList<EntityMarker> Entities => entities;

        private void Awake()
        {
            // Auto-find player if not set
            if (playerTransform == null)
            {
                GameObject playerObj = GameObject.FindGameObjectWithTag("Player");
                if (playerObj != null)
                {
                    playerTransform = playerObj.transform;
                }
            }

            // Auto-find camera if not set
            if (playerCamera == null)
            {
                playerCamera = Camera.main;
            }
        }

        private void Start()
        {
            // Discover all rooms and entities in scene
            DiscoverWorldObjects();

            // Validate scene setup — warn about common misconfigurations
            if (rooms.Count == 0)
            {
                Debug.LogWarning("[WorldManager] No RoomVolume objects found in scene. " +
                    "current_room_guid will always be empty. Add RoomVolume components to define spatial regions.");
            }

            if (playerTransform != null)
            {
                var cc = playerTransform.GetComponent<CharacterController>();
                var col = playerTransform.GetComponent<Collider>();
                bool hasTag = playerTransform.CompareTag("Player");
                if (cc == null && col == null && !hasTag)
                {
                    Debug.LogWarning("[WorldManager] Player has no CharacterController, Collider, or 'Player' tag. " +
                        "RoomVolume triggers will not detect the player.");
                }
            }
        }

        private void OnDestroy()
        {
            // Unsubscribe from all events
            foreach (var room in rooms)
            {
                if (room != null)
                {
                    room.OnPlayerEntered -= HandlePlayerEnteredRoom;
                    room.OnPlayerExited -= HandlePlayerExitedRoom;
                }
            }

            foreach (var entity in entities)
            {
                if (entity != null)
                {
                    var interactable = entity.GetComponent<InteractableState>();
                    if (interactable != null)
                    {
                        interactable.OnStateChanged -= HandleEntityStateChanged;
                    }
                }
            }
        }

        /// <summary>
        /// Discover all RoomVolumes and EntityMarkers in the scene.
        /// </summary>
        public void DiscoverWorldObjects()
        {
            // Clear existing
            rooms.Clear();
            entities.Clear();

            // Find all rooms
            RoomVolume[] foundRooms = FindObjectsByType<RoomVolume>(FindObjectsSortMode.None);
            foreach (var room in foundRooms)
            {
                RegisterRoom(room);
            }

            // Find all entities
            EntityMarker[] foundEntities = FindObjectsByType<EntityMarker>(FindObjectsSortMode.None);
            foreach (var entity in foundEntities)
            {
                RegisterEntity(entity);
            }

            // Auto-discover entities from scene Renderers if none were found
            if (entities.Count == 0 && autoDiscoverEntities)
            {
                AutoDiscoverEntities();
            }

            Debug.Log($"[WorldManager] Discovered {rooms.Count} rooms and {entities.Count} entities");

            // Determine initial room
            DetermineCurrentRoom();
        }

        /// <summary>
        /// Register a room with the world manager.
        /// </summary>
        public void RegisterRoom(RoomVolume room)
        {
            if (room == null || rooms.Contains(room)) return;

            rooms.Add(room);
            room.OnPlayerEntered += HandlePlayerEnteredRoom;
            room.OnPlayerExited += HandlePlayerExitedRoom;

            if (debugLogging)
            {
                Debug.Log($"[WorldManager] Registered room: {room.Label}");
            }
        }

        /// <summary>
        /// Auto-discover scene objects as entities when no EntityMarker components exist.
        /// Finds Renderers in the scene and attaches EntityMarker to qualifying objects.
        /// </summary>
        private void AutoDiscoverEntities()
        {
            Renderer[] allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
            int discovered = 0;

            foreach (var renderer in allRenderers)
            {
                GameObject go = renderer.gameObject;

                // Skip objects on excluded layers
                if (autoDiscoverExcludeLayers != 0 &&
                    ((1 << go.layer) & autoDiscoverExcludeLayers.value) != 0)
                    continue;

                // Skip if it already has an EntityMarker
                if (go.GetComponent<EntityMarker>() != null)
                    continue;

                // Skip RoomVolume objects (they are regions, not entities)
                if (go.GetComponent<RoomVolume>() != null)
                    continue;

                // Skip the player object
                if (playerTransform != null &&
                    (go.transform == playerTransform || go.transform.IsChildOf(playerTransform)))
                    continue;

                // Skip particle systems, trail renderers, line renderers
                if (renderer is ParticleSystemRenderer || renderer is TrailRenderer || renderer is LineRenderer)
                    continue;

                // Skip objects that are too small (UI elements, decals, etc.)
                Vector3 size = renderer.bounds.size;
                float volume = size.x * size.y * size.z;
                if (volume < minEntityVolume)
                    continue;

                // Attach EntityMarker — GUID is auto-generated in EntityMarker.Awake()
                EntityMarker marker = go.AddComponent<EntityMarker>();
                RegisterEntity(marker);
                discovered++;

                if (debugLogging)
                {
                    Debug.Log($"[WorldManager] Auto-discovered entity: {go.name} (vol={volume:F3})");
                }
            }

            Debug.Log($"[WorldManager] Auto-discovered {discovered} entities from scene Renderers");
        }

        /// <summary>
        /// Register an entity with the world manager.
        /// </summary>
        public void RegisterEntity(EntityMarker entity)
        {
            if (entity == null || entities.Contains(entity)) return;

            entities.Add(entity);

            // Subscribe to state changes if interactable
            var interactable = entity.GetComponent<InteractableState>();
            if (interactable != null)
            {
                interactable.OnStateChanged += HandleEntityStateChanged;
            }

            if (debugLogging)
            {
                Debug.Log($"[WorldManager] Registered entity: {entity.Label}");
            }
        }

        /// <summary>
        /// Unregister an entity from the world manager.
        /// </summary>
        public void UnregisterEntity(EntityMarker entity)
        {
            if (entity == null) return;

            entities.Remove(entity);

            var interactable = entity.GetComponent<InteractableState>();
            if (interactable != null)
            {
                interactable.OnStateChanged -= HandleEntityStateChanged;
            }
        }

        /// <summary>
        /// Get room by GUID.
        /// </summary>
        public RoomVolume GetRoomByGuid(string guid)
        {
            return rooms.Find(r => r.Guid == guid);
        }

        /// <summary>
        /// Get room by label.
        /// </summary>
        public RoomVolume GetRoomByLabel(string label)
        {
            return rooms.Find(r => r.Label.Equals(label, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Get entity by GUID.
        /// </summary>
        public EntityMarker GetEntityByGuid(string guid)
        {
            return entities.Find(e => e.Guid == guid);
        }

        /// <summary>
        /// Get entity by label.
        /// </summary>
        public EntityMarker GetEntityByLabel(string label)
        {
            return entities.Find(e => e.Label.Equals(label, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Get entities in a specific room.
        /// </summary>
        public List<EntityMarker> GetEntitiesInRoom(RoomVolume room)
        {
            if (room == null) return new List<EntityMarker>();

            List<EntityMarker> result = new List<EntityMarker>();
            foreach (var entity in entities)
            {
                if (room.ContainsPoint(entity.transform.position))
                {
                    result.Add(entity);
                }
            }
            return result;
        }

        /// <summary>
        /// Get entities visible from player camera.
        /// </summary>
        public List<EntityMarker> GetVisibleEntities()
        {
            List<EntityMarker> result = new List<EntityMarker>();
            foreach (var entity in entities)
            {
                if (entity.IsVisibleFrom(playerCamera))
                {
                    result.Add(entity);
                }
            }
            return result;
        }

        /// <summary>
        /// Get the next frame ID (monotonically increasing).
        /// </summary>
        public long GetNextFrameId()
        {
            return ++frameIdCounter;
        }

        /// <summary>
        /// Get and clear pending state changes since last call.
        /// </summary>
        public List<StateChangeEvent> GetAndClearStateChanges()
        {
            List<StateChangeEvent> changes = new List<StateChangeEvent>(pendingStateChanges);
            pendingStateChanges.Clear();
            return changes;
        }

        #region Ball Spawning

        /// <summary>
        /// Spawn a ball at the given position.
        /// </summary>
        public bool SpawnBall(Vector3 position)
        {
            if (spawnedBall != null)
            {
                // Already spawned, just move it
                return MoveBall(position);
            }

            if (ballPrefab == null)
            {
                Debug.LogWarning("[WorldManager] Ball prefab not assigned");
                return false;
            }

            spawnedBall = Instantiate(ballPrefab, position, Quaternion.identity);
            spawnedBall.name = "SpawnedBall";

            // Ensure it has an EntityMarker
            spawnedBallMarker = spawnedBall.GetComponent<EntityMarker>();
            if (spawnedBallMarker == null)
            {
                spawnedBallMarker = spawnedBall.AddComponent<EntityMarker>();
            }

            RegisterEntity(spawnedBallMarker);

            // Record state change
            RecordStateChange(spawnedBallMarker.Guid, "spawned", "none", "exists");

            return true;
        }

        /// <summary>
        /// Despawn the ball.
        /// </summary>
        public bool DespawnBall()
        {
            if (spawnedBall == null)
            {
                return false;
            }

            string guid = spawnedBallMarker?.Guid ?? "unknown";
            
            UnregisterEntity(spawnedBallMarker);
            Destroy(spawnedBall);
            spawnedBall = null;
            spawnedBallMarker = null;

            // Record state change
            RecordStateChange(guid, "despawned", "exists", "none");

            return true;
        }

        /// <summary>
        /// Move the ball to a new position.
        /// </summary>
        public bool MoveBall(Vector3 position)
        {
            if (spawnedBall == null)
            {
                return false;
            }

            spawnedBall.transform.position = position;
            return true;
        }

        #endregion

        #region World Reset

        /// <summary>
        /// Reset the world to initial state.
        /// </summary>
        public void ResetWorld()
        {
            // Despawn ball if exists
            DespawnBall();

            // Reset all interactables to initial state
            foreach (var entity in entities)
            {
                var interactable = entity.GetComponent<InteractableState>();
                if (interactable != null)
                {
                    interactable.SetStateIndex(0);
                }
            }

            // Clear pending changes
            pendingStateChanges.Clear();

            Debug.Log("[WorldManager] World reset complete");
        }

        #endregion

        #region Event Handlers

        private void HandlePlayerEnteredRoom(Transform player)
        {
            // Find which room
            foreach (var room in rooms)
            {
                if (room.IsPlayerInRoom)
                {
                    RoomVolume oldRoom = CurrentRoom;
                    CurrentRoom = room;

                    // Record state change
                    RecordStateChange("player", "room_changed", 
                        oldRoom?.Guid ?? "none", room.Guid);

                    OnRoomEntered?.Invoke(room);
                    break;
                }
            }
        }

        private void HandlePlayerExitedRoom(Transform player)
        {
            // Check if player is still in any room
            RoomVolume stillInRoom = null;
            foreach (var room in rooms)
            {
                if (room.IsPlayerInRoom)
                {
                    stillInRoom = room;
                    break;
                }
            }

            if (stillInRoom == null)
            {
                RoomVolume oldRoom = CurrentRoom;
                CurrentRoom = null;
                OnRoomExited?.Invoke(oldRoom);
            }
        }

        private void HandleEntityStateChanged(InteractableState interactable, string oldState, string newState)
        {
            EntityMarker marker = interactable.GetComponent<EntityMarker>();
            if (marker == null) return;

            RecordStateChange(marker.Guid, "state_changed", oldState, newState);
            OnEntityStateChanged?.Invoke(marker, oldState, newState);
        }

        private void RecordStateChange(string entityGuid, string changeType, string oldValue, string newValue)
        {
            StateChangeEvent change = new StateChangeEvent
            {
                entity_guid = entityGuid,
                change_type = changeType,
                old_value = oldValue,
                new_value = newValue,
                timestamp = DateTime.UtcNow.ToString("o")
            };

            pendingStateChanges.Add(change);

            if (debugLogging)
            {
                Debug.Log($"[WorldManager] State change: {entityGuid} {changeType}: {oldValue} -> {newValue}");
            }
        }

        private void DetermineCurrentRoom()
        {
            if (playerTransform == null) return;

            foreach (var room in rooms)
            {
                if (room.ContainsPoint(playerTransform.position))
                {
                    CurrentRoom = room;
                    Debug.Log($"[WorldManager] Player starts in room: {room.Label}");
                    return;
                }
            }

            Debug.Log("[WorldManager] Player is not in any room");
        }

        #endregion
    }
}

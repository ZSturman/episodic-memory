using System;
using UnityEngine;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Marks a trigger volume as a "room" with a stable GUID and label.
    /// The player entering this volume is considered to be in this room.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class RoomVolume : MonoBehaviour
    {
        [Header("Room Identity")]
        [SerializeField] private string roomGuid;
        [SerializeField] private string label = "Unnamed Room";
        [SerializeField] private Color gizmoColor = new Color(0f, 1f, 0f, 0.25f);

        [Header("Spawn Point")]
        [SerializeField] private Transform spawnPoint;  // Optional custom spawn point

        // Public accessors
        public string Guid => roomGuid;
        public string Label => label;

        // Events
        public event Action<Transform> OnPlayerEntered;
        public event Action<Transform> OnPlayerExited;

        // Track if player is currently in this room
        private bool playerInRoom;
        public bool IsPlayerInRoom => playerInRoom;

        private void Awake()
        {
            // Generate GUID if not set
            if (string.IsNullOrEmpty(roomGuid))
            {
                roomGuid = System.Guid.NewGuid().ToString();
                Debug.Log($"[RoomVolume] Generated new GUID for room '{label}': {roomGuid}");
            }

            // Ensure collider is a trigger
            Collider col = GetComponent<Collider>();
            if (col != null && !col.isTrigger)
            {
                col.isTrigger = true;
                Debug.LogWarning($"[RoomVolume] Set collider to trigger on room '{label}'");
            }
        }

        private void OnValidate()
        {
            // Auto-generate GUID in editor if empty
            if (string.IsNullOrEmpty(roomGuid))
            {
                roomGuid = System.Guid.NewGuid().ToString();
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            if (IsPlayer(other))
            {
                playerInRoom = true;
                Debug.Log($"[RoomVolume] Player entered room: {label}");
                OnPlayerEntered?.Invoke(other.transform);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (IsPlayer(other))
            {
                playerInRoom = false;
                Debug.Log($"[RoomVolume] Player exited room: {label}");
                OnPlayerExited?.Invoke(other.transform);
            }
        }

        private bool IsPlayer(Collider other)
        {
            // Check by tag or layer - customize as needed
            return other.CompareTag("Player") || 
                   other.GetComponent<CharacterController>() != null;
        }

        /// <summary>
        /// Get a safe spawn point within this room.
        /// </summary>
        public Vector3 GetSpawnPoint()
        {
            if (spawnPoint != null)
            {
                return spawnPoint.position;
            }

            // Default to center of room, slightly above floor
            return transform.position + Vector3.up * 0.5f;
        }

        /// <summary>
        /// Get bounds of this room volume.
        /// </summary>
        public Bounds GetBounds()
        {
            Collider col = GetComponent<Collider>();
            return col != null ? col.bounds : new Bounds(transform.position, Vector3.one);
        }

        /// <summary>
        /// Check if a world position is inside this room.
        /// </summary>
        public bool ContainsPoint(Vector3 worldPoint)
        {
            Collider col = GetComponent<Collider>();
            if (col == null) return false;

            // Use bounds check (approximation for non-box colliders)
            return col.bounds.Contains(worldPoint);
        }

#if UNITY_EDITOR
        private void OnDrawGizmos()
        {
            Collider col = GetComponent<Collider>();
            if (col == null) return;

            Gizmos.color = gizmoColor;
            
            if (col is BoxCollider box)
            {
                Gizmos.matrix = transform.localToWorldMatrix;
                Gizmos.DrawCube(box.center, box.size);
                Gizmos.color = new Color(gizmoColor.r, gizmoColor.g, gizmoColor.b, 1f);
                Gizmos.DrawWireCube(box.center, box.size);
            }
            else
            {
                Gizmos.DrawCube(col.bounds.center, col.bounds.size);
            }

            // Draw spawn point
            Vector3 spawn = GetSpawnPoint();
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(spawn, 0.3f);
        }

        private void OnDrawGizmosSelected()
        {
            // Draw label
            UnityEditor.Handles.Label(transform.position + Vector3.up * 2f, 
                $"Room: {label}\nGUID: {(roomGuid?.Substring(0, 8) ?? "none")}...");
        }
#endif
    }
}

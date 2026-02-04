using System;
using UnityEngine;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Marks a GameObject as a trackable entity with stable GUID.
    /// 
    /// ARCHITECTURAL INVARIANT: Unity does not assign semantic labels.
    /// All labels are owned by the Python backend and learned from user interaction.
    /// This component provides only observable properties: GUID, position, size, state.
    /// </summary>
    public class EntityMarker : MonoBehaviour
    {
        [Header("Entity Identity")]
        [SerializeField] private string entityGuid;
        
        // REMOVED: label and category fields - backend owns all semantic labeling
        // Labels are learned from user interaction, not predefined in Unity.

        [Header("Tracking")]
        [SerializeField] private bool trackPosition = true;
        [SerializeField] private bool trackRotation = true;
        [SerializeField] private float significantMoveThreshold = 0.1f;  // Units

        // Public accessors
        public string Guid => entityGuid;
        // REMOVED: Label and Category properties - backend owns semantic meaning

        // Track last known transform for change detection
        private Vector3 lastPosition;
        private Quaternion lastRotation;

        // Events
        public event Action<EntityMarker> OnEntityMoved;
        public event Action<EntityMarker> OnEntityRotated;

        private void Awake()
        {
            // Generate GUID if not set
            if (string.IsNullOrEmpty(entityGuid))
            {
                entityGuid = System.Guid.NewGuid().ToString();
                Debug.Log($"[EntityMarker] Generated new GUID: {entityGuid}");
            }

            lastPosition = transform.position;
            lastRotation = transform.rotation;
        }

        private void OnValidate()
        {
            // Auto-generate GUID in editor if empty
            if (string.IsNullOrEmpty(entityGuid))
            {
                entityGuid = System.Guid.NewGuid().ToString();
            }
        }

        private void Update()
        {
            if (trackPosition)
            {
                CheckPositionChange();
            }

            if (trackRotation)
            {
                CheckRotationChange();
            }
        }

        private void CheckPositionChange()
        {
            float distance = Vector3.Distance(transform.position, lastPosition);
            if (distance > significantMoveThreshold)
            {
                lastPosition = transform.position;
                OnEntityMoved?.Invoke(this);
            }
        }

        private void CheckRotationChange()
        {
            float angle = Quaternion.Angle(transform.rotation, lastRotation);
            if (angle > 5f)  // 5 degree threshold
            {
                lastRotation = transform.rotation;
                OnEntityRotated?.Invoke(this);
            }
        }

        /// <summary>
        /// Get the current position of this entity.
        /// </summary>
        public Vector3 GetPosition()
        {
            return transform.position;
        }

        /// <summary>
        /// Get the current rotation of this entity as euler angles.
        /// </summary>
        public Vector3 GetRotation()
        {
            return transform.eulerAngles;
        }

        /// <summary>
        /// Get the current state as a string (for InteractableState or custom state).
        /// Override in subclasses for custom state reporting.
        /// </summary>
        public virtual string GetStateString()
        {
            // Check if this entity has an InteractableState component
            InteractableState interactable = GetComponent<InteractableState>();
            if (interactable != null)
            {
                return interactable.CurrentState;
            }

            return "default";
        }

        /// <summary>
        /// Check if this entity is visible from a given camera.
        /// </summary>
        public bool IsVisibleFrom(Camera camera)
        {
            if (camera == null) return true;  // Assume visible if no camera

            // Get renderer bounds or use position
            Renderer renderer = GetComponent<Renderer>();
            if (renderer != null)
            {
                Plane[] planes = GeometryUtility.CalculateFrustumPlanes(camera);
                return GeometryUtility.TestPlanesAABB(planes, renderer.bounds);
            }

            // Fallback: check if position is in viewport
            Vector3 viewportPoint = camera.WorldToViewportPoint(transform.position);
            return viewportPoint.x >= 0 && viewportPoint.x <= 1 &&
                   viewportPoint.y >= 0 && viewportPoint.y <= 1 &&
                   viewportPoint.z > 0;  // In front of camera
        }

        /// <summary>
        /// Get approximate distance from a point (useful for relevance sorting).
        /// </summary>
        public float DistanceFrom(Vector3 point)
        {
            return Vector3.Distance(transform.position, point);
        }

#if UNITY_EDITOR
        private void OnDrawGizmos()
        {
            // Draw entity marker icon - neutral color since backend owns semantics
            Gizmos.color = Color.white;
            Gizmos.DrawSphere(transform.position + Vector3.up * 0.1f, 0.15f);
        }

        private void OnDrawGizmosSelected()
        {
            // Draw GUID only - labels are backend-owned
            UnityEditor.Handles.Label(transform.position + Vector3.up * 0.5f, 
                $"Entity\nGUID: {(entityGuid?.Substring(0, 8) ?? "none")}...");
        }
#endif
    }
}

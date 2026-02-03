using System;
using UnityEngine;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Marks a GameObject as a trackable entity with stable GUID, label, and optional category.
    /// This is the primary component for anything the agent should be aware of.
    /// </summary>
    public class EntityMarker : MonoBehaviour
    {
        [Header("Entity Identity")]
        [SerializeField] private string entityGuid;
        [SerializeField] private string label = "Unnamed Entity";
        [SerializeField] private string category = "object";  // e.g., "furniture", "item", "prop", "door"

        [Header("Tracking")]
        [SerializeField] private bool trackPosition = true;
        [SerializeField] private bool trackRotation = true;
        [SerializeField] private float significantMoveThreshold = 0.1f;  // Units

        // Public accessors
        public string Guid => entityGuid;
        public string Label => label;
        public string Category => category;

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
                Debug.Log($"[EntityMarker] Generated new GUID for entity '{label}': {entityGuid}");
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
            // Draw entity marker icon
            Gizmos.color = GetCategoryColor();
            Gizmos.DrawSphere(transform.position + Vector3.up * 0.1f, 0.15f);
        }

        private void OnDrawGizmosSelected()
        {
            // Draw label when selected
            UnityEditor.Handles.Label(transform.position + Vector3.up * 0.5f, 
                $"{label}\n[{category}]\n{(entityGuid?.Substring(0, 8) ?? "none")}...");
        }

        private Color GetCategoryColor()
        {
            return category switch
            {
                "furniture" => Color.yellow,
                "door" => Color.cyan,
                "item" => Color.green,
                "prop" => Color.magenta,
                _ => Color.white
            };
        }
#endif
    }
}

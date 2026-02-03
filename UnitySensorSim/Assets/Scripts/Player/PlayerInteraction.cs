using System;
using UnityEngine;
using EpisodicAgent.World;

namespace EpisodicAgent.Player
{
    /// <summary>
    /// Handles player interaction with interactable objects.
    /// Uses raycasting to detect and interact with InteractableState components.
    /// </summary>
    public class PlayerInteraction : MonoBehaviour
    {
        [Header("Interaction Settings")]
        [SerializeField] private float interactionRange = 3f;
        [SerializeField] private KeyCode interactKey = KeyCode.E;
        [SerializeField] private LayerMask interactableMask = -1;  // Everything by default

        [Header("References")]
        [SerializeField] private Camera playerCamera;

        [Header("UI Feedback")]
        [SerializeField] private bool showInteractionPrompt = true;
        [SerializeField] private string promptFormat = "Press {0} to interact with {1}";

        // Events
        public event Action<EntityMarker, InteractableState> OnInteraction;
        public event Action<EntityMarker> OnLookAtInteractable;
        public event Action OnLookAway;

        // Current target
        private InteractableState currentTarget;
        private EntityMarker currentTargetEntity;
        private string interactionPrompt = "";

        public InteractableState CurrentTarget => currentTarget;
        public EntityMarker CurrentTargetEntity => currentTargetEntity;
        public string InteractionPrompt => interactionPrompt;
        public bool HasTarget => currentTarget != null;

        private void Awake()
        {
            if (playerCamera == null)
            {
                playerCamera = Camera.main;
            }
        }

        private void Update()
        {
            CheckForInteractable();
            HandleInteraction();
        }

        private void CheckForInteractable()
        {
            if (playerCamera == null) return;

            Ray ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, interactionRange, interactableMask))
            {
                // Check if hit object has InteractableState
                InteractableState interactable = hit.collider.GetComponent<InteractableState>();
                if (interactable == null)
                {
                    interactable = hit.collider.GetComponentInParent<InteractableState>();
                }

                if (interactable != null)
                {
                    EntityMarker entity = interactable.GetComponent<EntityMarker>();
                    
                    // New target
                    if (currentTarget != interactable)
                    {
                        currentTarget = interactable;
                        currentTargetEntity = entity;
                        
                        UpdatePrompt();
                        OnLookAtInteractable?.Invoke(entity);
                    }
                    return;
                }
            }

            // No target
            if (currentTarget != null)
            {
                currentTarget = null;
                currentTargetEntity = null;
                interactionPrompt = "";
                OnLookAway?.Invoke();
            }
        }

        private void HandleInteraction()
        {
            if (currentTarget == null) return;

            if (Input.GetKeyDown(interactKey))
            {
                Interact();
            }
        }

        /// <summary>
        /// Interact with the current target.
        /// </summary>
        public void Interact()
        {
            if (currentTarget == null) return;

            string oldState = currentTarget.CurrentState;
            currentTarget.Toggle();
            string newState = currentTarget.CurrentState;

            Debug.Log($"[PlayerInteraction] Interacted with {currentTargetEntity?.Label ?? "unknown"}: {oldState} -> {newState}");

            OnInteraction?.Invoke(currentTargetEntity, currentTarget);
            UpdatePrompt();
        }

        /// <summary>
        /// Force interact with a specific interactable (for testing/commands).
        /// </summary>
        public void InteractWith(InteractableState interactable)
        {
            if (interactable == null) return;

            EntityMarker entity = interactable.GetComponent<EntityMarker>();
            
            string oldState = interactable.CurrentState;
            interactable.Toggle();
            string newState = interactable.CurrentState;

            Debug.Log($"[PlayerInteraction] Force interacted with {entity?.Label ?? "unknown"}: {oldState} -> {newState}");

            OnInteraction?.Invoke(entity, interactable);
        }

        private void UpdatePrompt()
        {
            if (!showInteractionPrompt || currentTarget == null)
            {
                interactionPrompt = "";
                return;
            }

            string entityName = currentTargetEntity?.Label ?? "object";
            string keyName = interactKey.ToString();
            string stateInfo = $"[{currentTarget.CurrentState}]";

            interactionPrompt = string.Format(promptFormat, keyName, entityName) + " " + stateInfo;
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            if (playerCamera == null) return;

            // Draw interaction ray
            Gizmos.color = HasTarget ? Color.green : Color.yellow;
            Gizmos.DrawRay(playerCamera.transform.position, 
                playerCamera.transform.forward * interactionRange);
        }
#endif
    }
}

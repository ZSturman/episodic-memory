using System;
using UnityEngine;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Exposes interactable state like Open/Closed, On/Off for entities.
    /// Attach to any EntityMarker that has toggleable state.
    /// </summary>
    [RequireComponent(typeof(EntityMarker))]
    public class InteractableState : MonoBehaviour
    {
        public enum StateType
        {
            OnOff,      // On, Off
            OpenClosed, // Open, Closed
            Custom      // Uses customStates array
        }

        [Header("State Configuration")]
        [SerializeField] private StateType stateType = StateType.OnOff;
        [SerializeField] private int currentStateIndex;
        [SerializeField] private string[] customStates = { "State1", "State2" };

        [Header("Visuals (Optional)")]
        [SerializeField] private GameObject[] stateObjects;  // Objects to show/hide per state
        [SerializeField] private Animator animator;
        [SerializeField] private string animatorParameter = "StateIndex";

        // Events
        public event Action<InteractableState, string, string> OnStateChanged;  // interactable, oldState, newState

        // Public accessors
        public string CurrentState => GetStateAtIndex(currentStateIndex);
        public int CurrentStateIndex => currentStateIndex;
        public int StateCount => GetStateCount();
        public StateType Type => stateType;

        private EntityMarker entityMarker;

        private void Awake()
        {
            entityMarker = GetComponent<EntityMarker>();
            
            // Auto-find animator if not set
            if (animator == null)
            {
                animator = GetComponent<Animator>();
            }

            // Apply initial state visuals
            ApplyStateVisuals();
        }

        /// <summary>
        /// Toggle to next state (cycles through states).
        /// </summary>
        public void Toggle()
        {
            int nextIndex = (currentStateIndex + 1) % GetStateCount();
            SetStateIndex(nextIndex);
        }

        /// <summary>
        /// Set state by name.
        /// </summary>
        public void SetState(string stateName)
        {
            int index = GetIndexForState(stateName);
            if (index >= 0)
            {
                SetStateIndex(index);
            }
            else
            {
                Debug.LogWarning($"[InteractableState] Unknown state: {stateName} for {entityMarker?.Label}");
            }
        }

        /// <summary>
        /// Set state by index.
        /// </summary>
        public void SetStateIndex(int index)
        {
            if (index < 0 || index >= GetStateCount())
            {
                Debug.LogWarning($"[InteractableState] Invalid state index: {index}");
                return;
            }

            string oldState = CurrentState;
            currentStateIndex = index;
            string newState = CurrentState;

            if (oldState != newState)
            {
                Debug.Log($"[InteractableState] {entityMarker?.Label}: {oldState} -> {newState}");
                ApplyStateVisuals();
                OnStateChanged?.Invoke(this, oldState, newState);
            }
        }

        /// <summary>
        /// Check if currently in a specific state.
        /// </summary>
        public bool IsInState(string stateName)
        {
            return CurrentState.Equals(stateName, StringComparison.OrdinalIgnoreCase);
        }

        private string GetStateAtIndex(int index)
        {
            return stateType switch
            {
                StateType.OnOff => index == 0 ? "Off" : "On",
                StateType.OpenClosed => index == 0 ? "Closed" : "Open",
                StateType.Custom => (index >= 0 && index < customStates.Length) 
                    ? customStates[index] : "Unknown",
                _ => "Unknown"
            };
        }

        private int GetIndexForState(string stateName)
        {
            string lower = stateName.ToLowerInvariant();
            
            switch (stateType)
            {
                case StateType.OnOff:
                    if (lower == "off" || lower == "false" || lower == "0") return 0;
                    if (lower == "on" || lower == "true" || lower == "1") return 1;
                    break;

                case StateType.OpenClosed:
                    if (lower == "closed" || lower == "close" || lower == "false" || lower == "0") return 0;
                    if (lower == "open" || lower == "true" || lower == "1") return 1;
                    break;

                case StateType.Custom:
                    for (int i = 0; i < customStates.Length; i++)
                    {
                        if (customStates[i].Equals(stateName, StringComparison.OrdinalIgnoreCase))
                        {
                            return i;
                        }
                    }
                    break;
            }

            return -1;
        }

        private int GetStateCount()
        {
            return stateType switch
            {
                StateType.OnOff => 2,
                StateType.OpenClosed => 2,
                StateType.Custom => customStates.Length,
                _ => 2
            };
        }

        private void ApplyStateVisuals()
        {
            // Update state objects visibility
            if (stateObjects != null && stateObjects.Length > 0)
            {
                for (int i = 0; i < stateObjects.Length; i++)
                {
                    if (stateObjects[i] != null)
                    {
                        stateObjects[i].SetActive(i == currentStateIndex);
                    }
                }
            }

            // Update animator
            if (animator != null)
            {
                animator.SetInteger(animatorParameter, currentStateIndex);
            }
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            // Draw state info
            string label = entityMarker != null ? entityMarker.Label : gameObject.name;
            UnityEditor.Handles.Label(transform.position + Vector3.up * 1f, 
                $"[Interactable]\n{label}\nState: {CurrentState}");
        }
#endif
    }
}

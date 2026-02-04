# Unity UI System

This document describes the UI architecture for the Unity sensor simulator.

> **ARCHITECTURAL INVARIANT:** Unity is stateless. UI displays only what backend sends. Unity does NOT interpret, cache, or modify labels.

---

## UI Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Location: [Unknown]              Entities: 5 detected   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│                                                                  │
│                    ┌─────────────────────┐                      │
│                    │                     │                      │
│                    │  [Entity: Unknown]  │  <- Floating label   │
│                    │                     │                      │
│                    └─────────────────────┘                      │
│                                                                  │
│                                                                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ [!] Backend is asking: "What is this object?"           │    │ <- Label Request
│  │                                                          │    │
│  │ [Input: ___________________]  [Submit] [Skip]           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Components

### 1. Status Bar (Top)

Shows current backend state:

| Field | Source | Default |
|-------|--------|---------|
| Location | `location_update.label` | "Unknown" |
| Confidence | `location_update.confidence` | 0% |
| Entity Count | Count of `entity_update` messages | 0 |
| Stability | `location_update.is_stable` | "Uncertain" |

**Implementation:**

```csharp
public class StatusBarUI : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI locationText;
    [SerializeField] private TextMeshProUGUI confidenceText;
    [SerializeField] private TextMeshProUGUI entityCountText;
    [SerializeField] private Image stabilityIndicator;

    // Called when backend sends location_update
    public void OnLocationUpdate(LocationUpdate update)
    {
        locationText.text = $"Location: {update.label}";
        confidenceText.text = $"{update.confidence:P0}";
        stabilityIndicator.color = update.is_stable ? Color.green : Color.yellow;
    }
}
```

### 2. Entity Labels (Floating)

3D labels that float above detected entities:

**Behavior:**
- Default: "Unknown" (gray)
- After label: User's label (white)
- Confidence shown if < 100%

**INVARIANT:** Labels come ONLY from `entity_update` messages.

```csharp
public class EntityLabelUI : MonoBehaviour
{
    [SerializeField] private TextMeshPro labelText;
    [SerializeField] private string entityGuid;

    private void Start()
    {
        // Start with "Unknown" - no assumptions
        labelText.text = "Unknown";
        labelText.color = Color.gray;
    }

    // Called when backend sends entity_update for this entity
    public void OnEntityUpdate(EntityUpdate update)
    {
        if (update.entity_id != entityGuid) return;

        labelText.text = update.label;
        labelText.color = update.confidence > 0.8f ? Color.white : Color.yellow;
    }
}
```

### 3. Label Request Modal

Modal dialog when backend needs user input:

```
┌──────────────────────────────────────────────┐
│  What is this?                               │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │ [Thumbnail or highlight in scene]      │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Suggestions: [mug] [cup] [container]        │
│                                              │
│  Your label: [____________________]          │
│                                              │
│  [Submit]  [Skip]                            │
└──────────────────────────────────────────────┘
```

**CRITICAL:** Suggestions come from backend only. Unity does NOT add suggestions based on scene data.

```csharp
public class LabelRequestModal : MonoBehaviour
{
    [SerializeField] private GameObject modalPanel;
    [SerializeField] private TextMeshProUGUI promptText;
    [SerializeField] private TMP_InputField labelInput;
    [SerializeField] private Transform suggestionsContainer;
    [SerializeField] private Button submitButton;
    [SerializeField] private Button skipButton;

    private string _currentRequestId;
    private System.Action<string, string> _onResponse;

    public void Show(LabelRequest request, System.Action<string, string> onResponse)
    {
        _currentRequestId = request.request_id;
        _onResponse = onResponse;

        promptText.text = request.description;
        labelInput.text = request.current_label ?? "";

        // Show suggestions from BACKEND ONLY
        ClearSuggestions();
        foreach (var suggestion in request.alternative_labels)
        {
            CreateSuggestionButton(suggestion);
        }

        modalPanel.SetActive(true);
        labelInput.Select();
    }

    private void CreateSuggestionButton(string label)
    {
        // Create button that fills input with suggestion
        var btn = Instantiate(suggestionPrefab, suggestionsContainer);
        btn.GetComponentInChildren<TextMeshProUGUI>().text = label;
        btn.onClick.AddListener(() => labelInput.text = label);
    }

    public void OnSubmit()
    {
        string label = labelInput.text.Trim();
        if (string.IsNullOrEmpty(label)) return;

        _onResponse?.Invoke(_currentRequestId, label);
        Hide();
    }

    public void OnSkip()
    {
        _onResponse?.Invoke(_currentRequestId, null);  // Skipped
        Hide();
    }

    private void Hide()
    {
        modalPanel.SetActive(false);
        _currentRequestId = null;
    }
}
```

---

## Message Handlers

### UIMessageHandler.cs

Central handler for all UI-related backend messages:

```csharp
using UnityEngine;
using EpisodicAgent.Protocol;

namespace EpisodicAgent.UI
{
    /// <summary>
    /// Routes backend messages to appropriate UI components.
    /// INVARIANT: No caching. No interpretation. Just display.
    /// </summary>
    public class UIMessageHandler : MonoBehaviour
    {
        [SerializeField] private StatusBarUI statusBar;
        [SerializeField] private EntityLabelManager entityLabels;
        [SerializeField] private LabelRequestModal labelModal;
        [SerializeField] private WebSocketClient webSocketClient;

        private void Start()
        {
            // Subscribe to message types
            webSocketClient.OnEntityUpdate += HandleEntityUpdate;
            webSocketClient.OnLocationUpdate += HandleLocationUpdate;
            webSocketClient.OnLabelRequest += HandleLabelRequest;
        }

        private void HandleEntityUpdate(EntityUpdate update)
        {
            // Forward to entity label system
            entityLabels.UpdateEntity(update);
        }

        private void HandleLocationUpdate(LocationUpdate update)
        {
            // Forward to status bar
            statusBar.OnLocationUpdate(update);
        }

        private void HandleLabelRequest(LabelRequest request)
        {
            // Show modal, send response when user submits
            labelModal.Show(request, (requestId, label) =>
            {
                var response = new LabelResponse
                {
                    request_id = requestId,
                    response_type = label != null ? "provided" : "skipped",
                    label = label
                };
                webSocketClient.SendLabelResponse(response);
            });
        }
    }
}
```

---

## Default Values

Everything starts as "Unknown" until backend provides information:

| Element | Default Display |
|---------|-----------------|
| Location label | "Unknown" |
| Entity labels | "Unknown" |
| Location confidence | 0% |
| Entity confidence | 0% |
| Stability indicator | Yellow (uncertain) |

**WHY:** The system knows nothing at startup. All knowledge emerges from user labeling.

---

## Setup Instructions

### 1. Create UI Canvas

1. Hierarchy → UI → Canvas
2. Set Canvas Scaler:
   - UI Scale Mode: Scale With Screen Size
   - Reference Resolution: 1920 x 1080

### 2. Create Status Bar

1. Create Panel at top of canvas
2. Add `StatusBarUI` component
3. Create child Text elements for location, confidence, entity count
4. Assign references

### 3. Create Label Request Modal

1. Create Panel (centered, initially disabled)
2. Add `LabelRequestModal` component
3. Create child elements: prompt text, input field, buttons, suggestions container
4. Assign references

### 4. Create Entity Label Prefab

1. Create 3D Text (TextMeshPro - 3D)
2. Add `EntityLabelUI` component
3. Save as prefab
4. `EntityLabelManager` instantiates these for detected entities

### 5. Wire Up Message Handler

1. Add `UIMessageHandler` to GameManager
2. Assign all UI component references
3. Assign WebSocketClient reference

---

## Testing

### Verify Default State

1. Start Unity without Python backend
2. All labels should show "Unknown"
3. Location should show "Unknown"
4. No errors in console

### Verify Label Request Flow

1. Start backend, connect
2. Backend sends `label_request`
3. Modal appears with backend's prompt and suggestions
4. Type label, submit
5. Backend receives `label_response`
6. Backend sends `entity_update` with new label
7. Entity label updates in UI

### Verify No Local Caching

1. Label an entity
2. Disconnect backend
3. Reconnect backend
4. Entity should show "Unknown" again (backend owns state)

---

## Troubleshooting

### "Labels don't update"

- Check `UIMessageHandler` subscriptions
- Verify `EntityLabelManager` receives updates
- Check entity GUID matching

### "Modal doesn't appear"

- Check `LabelRequestModal` is active in hierarchy
- Verify panel is assigned and starts disabled
- Check WebSocket connection

### "Suggestions show scene data"

**BUG!** Unity should NEVER add suggestions based on scene analysis. Fix by ensuring `CreateSuggestionButton` only uses `request.alternative_labels`.

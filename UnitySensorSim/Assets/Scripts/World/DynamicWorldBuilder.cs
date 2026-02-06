using System;
using System.Collections.Generic;
using UnityEngine;
using EpisodicAgent.Protocol;

namespace EpisodicAgent.World
{
    /// <summary>
    /// Creates and manages visualization volumes at runtime based on
    /// commands from the Python backend.
    ///
    /// When the backend discovers a location boundary (via fingerprinting
    /// or any other resolver), it can send a create_room_volume command
    /// to Unity. This component creates a semi-transparent box/sphere
    /// volume in the scene to visualize the backend's spatial model.
    ///
    /// These dynamic volumes are purely for visualization — they do NOT
    /// affect the sensor stream or perception pipeline. The backend is
    /// the source of truth for all spatial decisions.
    /// </summary>
    public class DynamicWorldBuilder : MonoBehaviour
    {
        [Header("Defaults")]
        [SerializeField] private Color defaultColor = new Color(0.2f, 0.6f, 1f, 0.15f);
        [SerializeField] private Material volumeMaterial;

        [Header("Label Display")]
        [SerializeField] private Font labelFont;
        [SerializeField] private int labelFontSize = 14;

        // Tracking
        private Dictionary<string, DynamicVolume> volumes = new Dictionary<string, DynamicVolume>();
        private WorldManager worldManager;

        private void Start()
        {
            worldManager = FindFirstObjectByType<WorldManager>();

            // Create a default transparent material if none assigned
            if (volumeMaterial == null)
            {
                volumeMaterial = CreateTransparentMaterial();
            }
        }

        // =================================================================
        // PUBLIC API — called by CommandReceiver
        // =================================================================

        /// <summary>
        /// Create a new visualization volume for a backend-discovered location.
        /// </summary>
        public bool CreateVolume(string locationId, string label, Vector3 center,
                                  Vector3 extent, Color? color = null, float opacity = 0.15f)
        {
            if (string.IsNullOrEmpty(locationId))
            {
                Debug.LogWarning("[DynamicWorldBuilder] Missing locationId");
                return false;
            }

            // Remove existing if updating
            if (volumes.ContainsKey(locationId))
            {
                RemoveVolume(locationId);
            }

            // Create GameObject
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = $"DynVolume_{label}_{locationId.Substring(0, Mathf.Min(8, locationId.Length))}";
            go.transform.position = center;
            go.transform.localScale = extent * 2f; // extent = half-size

            // Disable collider so it doesn't interfere with gameplay
            Collider col = go.GetComponent<Collider>();
            if (col != null)
            {
                col.enabled = false;
            }

            // Apply transparent material
            Renderer rend = go.GetComponent<Renderer>();
            if (rend != null)
            {
                Material mat = new Material(volumeMaterial);
                Color c = color ?? defaultColor;
                c.a = opacity;
                mat.color = c;
                rend.material = mat;
            }

            // Create floating label
            GameObject labelObj = CreateFloatingLabel(label, center + Vector3.up * (extent.y + 0.5f));
            labelObj.transform.SetParent(go.transform);

            // Track it
            var volume = new DynamicVolume
            {
                LocationId = locationId,
                Label = label,
                Center = center,
                Extent = extent,
                GameObject = go,
                LabelObject = labelObj
            };
            volumes[locationId] = volume;

            Debug.Log($"[DynamicWorldBuilder] Created volume: {label} at {center} (extent: {extent})");
            return true;
        }

        /// <summary>
        /// Update an existing volume's properties.
        /// </summary>
        public bool UpdateVolume(string locationId, string newLabel = null,
                                  Vector3? newCenter = null, Vector3? newExtent = null,
                                  Color? newColor = null, float? newOpacity = null)
        {
            if (!volumes.TryGetValue(locationId, out DynamicVolume volume))
            {
                Debug.LogWarning($"[DynamicWorldBuilder] Volume not found: {locationId}");
                return false;
            }

            if (newCenter.HasValue)
            {
                volume.Center = newCenter.Value;
                volume.GameObject.transform.position = newCenter.Value;
            }

            if (newExtent.HasValue)
            {
                volume.Extent = newExtent.Value;
                volume.GameObject.transform.localScale = newExtent.Value * 2f;
            }

            if (newLabel != null)
            {
                volume.Label = newLabel;
                volume.GameObject.name = $"DynVolume_{newLabel}_{locationId.Substring(0, Mathf.Min(8, locationId.Length))}";
                UpdateFloatingLabel(volume.LabelObject, newLabel);
            }

            if (newColor.HasValue || newOpacity.HasValue)
            {
                Renderer rend = volume.GameObject.GetComponent<Renderer>();
                if (rend != null)
                {
                    Color c = newColor ?? rend.material.color;
                    if (newOpacity.HasValue) c.a = newOpacity.Value;
                    rend.material.color = c;
                }
            }

            return true;
        }

        /// <summary>
        /// Remove a specific dynamic volume.
        /// </summary>
        public bool RemoveVolume(string locationId)
        {
            if (!volumes.TryGetValue(locationId, out DynamicVolume volume))
            {
                return false;
            }

            if (volume.GameObject != null) Destroy(volume.GameObject);
            volumes.Remove(locationId);

            Debug.Log($"[DynamicWorldBuilder] Removed volume: {volume.Label}");
            return true;
        }

        /// <summary>
        /// Clear all dynamic volumes.
        /// </summary>
        public void ClearAllVolumes()
        {
            foreach (var kvp in volumes)
            {
                if (kvp.Value.GameObject != null)
                    Destroy(kvp.Value.GameObject);
            }
            volumes.Clear();
            Debug.Log("[DynamicWorldBuilder] Cleared all dynamic volumes");
        }

        /// <summary>
        /// Set a floating label on an existing entity by GUID.
        /// </summary>
        public bool SetEntityLabel(string entityGuid, string label)
        {
            if (worldManager == null)
            {
                Debug.LogWarning("[DynamicWorldBuilder] WorldManager not available");
                return false;
            }

            EntityMarker entity = worldManager.GetEntityByGuid(entityGuid);
            if (entity == null)
            {
                Debug.LogWarning($"[DynamicWorldBuilder] Entity not found: {entityGuid}");
                return false;
            }

            // Find or create a label child
            Transform existingLabel = entity.transform.Find("DynLabel");
            if (existingLabel != null)
            {
                UpdateFloatingLabel(existingLabel.gameObject, label);
            }
            else
            {
                Vector3 labelPos = entity.transform.position + Vector3.up * 1.5f;
                GameObject labelObj = CreateFloatingLabel(label, labelPos);
                labelObj.name = "DynLabel";
                labelObj.transform.SetParent(entity.transform);
                labelObj.transform.localPosition = Vector3.up * 1.5f;
            }

            Debug.Log($"[DynamicWorldBuilder] Set label '{label}' on entity {entityGuid}");
            return true;
        }

        // =================================================================
        // HELPERS
        // =================================================================

        private GameObject CreateFloatingLabel(string text, Vector3 position)
        {
            GameObject labelObj = new GameObject("FloatingLabel");
            labelObj.transform.position = position;

            // Use TextMesh for simple 3D text (works without UI Canvas)
            TextMesh textMesh = labelObj.AddComponent<TextMesh>();
            textMesh.text = text;
            textMesh.fontSize = labelFontSize;
            textMesh.characterSize = 0.1f;
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.alignment = TextAlignment.Center;
            textMesh.color = Color.white;
            if (labelFont != null) textMesh.font = labelFont;

            // Make it always face the camera
            labelObj.AddComponent<Billboard>();

            return labelObj;
        }

        private void UpdateFloatingLabel(GameObject labelObj, string text)
        {
            if (labelObj == null) return;
            TextMesh textMesh = labelObj.GetComponent<TextMesh>();
            if (textMesh != null) textMesh.text = text;
        }

        private static Material CreateTransparentMaterial()
        {
            // Use the built-in Standard shader with transparent rendering
            Shader shader = Shader.Find("Standard");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Color");
            }

            Material mat = new Material(shader);
            mat.SetFloat("_Mode", 3); // Transparent
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 3000;
            return mat;
        }

        /// <summary>
        /// Parse a hex color string like "#FF8800" into a Unity Color.
        /// </summary>
        public static Color ParseHexColor(string hex, Color fallback)
        {
            if (string.IsNullOrEmpty(hex)) return fallback;
            if (hex.StartsWith("#")) hex = hex.Substring(1);
            if (ColorUtility.TryParseHtmlString("#" + hex, out Color c))
                return c;
            return fallback;
        }

        // =================================================================
        // INNER TYPES
        // =================================================================

        private class DynamicVolume
        {
            public string LocationId;
            public string Label;
            public Vector3 Center;
            public Vector3 Extent;
            public GameObject GameObject;
            public GameObject LabelObject;
        }
    }

    /// <summary>
    /// Simple Billboard script — makes a GameObject always face the main camera.
    /// </summary>
    public class Billboard : MonoBehaviour
    {
        private Camera cam;

        private void Start()
        {
            cam = Camera.main;
        }

        private void LateUpdate()
        {
            if (cam != null)
            {
                transform.LookAt(transform.position + cam.transform.forward);
            }
        }
    }
}

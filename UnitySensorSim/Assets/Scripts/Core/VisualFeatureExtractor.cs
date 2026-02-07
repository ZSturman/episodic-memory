using System;
using System.Text;
using UnityEngine;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Captures the player camera view via RenderTexture and extracts per-cell
    /// visual features for the 4×4 FOV grid.
    ///
    /// ARCHITECTURAL INVARIANT: Unity = eyes, backend = brain.
    /// This component produces bandwidth-efficient feature summaries (color histograms,
    /// edge density, brightness, motion) — NOT object labels or identities.
    /// The Python backend interprets these features to discover and recognize entities.
    ///
    /// Feature extraction matches the Python schema (schemas/visual.py):
    ///   - 4×4 grid = 16 cells
    ///   - Per cell: color_histogram (24 bins: R×8, G×8, B×8), edge_directions (8 bins),
    ///               mean_brightness, edge_density, motion_magnitude
    ///   - Global: global_brightness, global_edge_density, global_motion,
    ///             dominant_colors (top 3 RGB), high_res_available
    /// </summary>
    public class VisualFeatureExtractor : MonoBehaviour
    {
        // =====================================================================
        // Configuration
        // =====================================================================

        [Header("Capture")]
        [SerializeField] private int captureWidth = 256;
        [SerializeField] private int captureHeight = 256;

        [Header("Debug")]
        [SerializeField] private bool debugLogging = false;

        // Grid constants matching Python (schemas/visual.py)
        private const int GRID_ROWS = 4;
        private const int GRID_COLS = 4;
        private const int GRID_CELL_COUNT = GRID_ROWS * GRID_COLS;
        private const int COLOR_HISTOGRAM_BINS = 8;  // Per channel
        private const int EDGE_DIRECTION_BINS = 8;

        // Capture resources
        private Camera _camera;
        private RenderTexture _renderTexture;
        private Texture2D _captureTexture;

        // Previous frame for motion detection
        private float[] _prevBrightness;

        // Reusable buffers (avoid GC per frame)
        private Color32[] _pixels;
        private readonly StringBuilder _jsonBuilder = new StringBuilder(8192);

        // =====================================================================
        // Lifecycle
        // =====================================================================

        private void Start()
        {
            // Find the player camera
            _camera = Camera.main;
            if (_camera == null)
            {
                Debug.LogWarning("[VisualFeatureExtractor] No Camera.main found. " +
                    "Visual features will not be extracted.");
                return;
            }

            // Create render texture for off-screen capture
            _renderTexture = new RenderTexture(captureWidth, captureHeight, 16, RenderTextureFormat.ARGB32);
            _renderTexture.Create();

            // CPU-side texture for pixel readback
            _captureTexture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

            // Motion detection buffer
            _prevBrightness = new float[GRID_CELL_COUNT];

            if (debugLogging)
                Debug.Log($"[VisualFeatureExtractor] Initialized {captureWidth}×{captureHeight} capture");
        }

        private void OnDestroy()
        {
            if (_renderTexture != null)
            {
                _renderTexture.Release();
                Destroy(_renderTexture);
            }
            if (_captureTexture != null)
                Destroy(_captureTexture);
        }

        // =====================================================================
        // Public API
        // =====================================================================

        /// <summary>
        /// Capture the current camera view and build a JSON summary string
        /// matching the Python VisualSummaryFrame schema.
        /// Returns null if capture fails.
        /// </summary>
        public string BuildSummaryJson(int frameId)
        {
            if (_camera == null || _renderTexture == null)
                return null;

            // Render to off-screen texture
            RenderTexture prevTarget = _camera.targetTexture;
            _camera.targetTexture = _renderTexture;
            _camera.Render();
            _camera.targetTexture = prevTarget;

            // Read pixels to CPU
            RenderTexture prevActive = RenderTexture.active;
            RenderTexture.active = _renderTexture;
            _captureTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0, false);
            _captureTexture.Apply();
            RenderTexture.active = prevActive;

            _pixels = _captureTexture.GetPixels32();

            // Extract per-cell features
            CellFeatures[] cells = new CellFeatures[GRID_CELL_COUNT];
            float globalBrightness = 0f;
            float globalEdgeDensity = 0f;
            float globalMotion = 0f;

            int cellW = captureWidth / GRID_COLS;
            int cellH = captureHeight / GRID_ROWS;

            for (int row = 0; row < GRID_ROWS; row++)
            {
                for (int col = 0; col < GRID_COLS; col++)
                {
                    int idx = row * GRID_COLS + col;
                    cells[idx] = ExtractCellFeatures(row, col, cellW, cellH);

                    globalBrightness += cells[idx].meanBrightness;
                    globalEdgeDensity += cells[idx].edgeDensity;

                    // Motion = absolute brightness change from previous frame
                    float motion = Mathf.Abs(cells[idx].meanBrightness - _prevBrightness[idx]);
                    cells[idx].motionMagnitude = motion;
                    globalMotion += motion;

                    _prevBrightness[idx] = cells[idx].meanBrightness;
                }
            }

            globalBrightness /= GRID_CELL_COUNT;
            globalEdgeDensity /= GRID_CELL_COUNT;
            globalMotion /= GRID_CELL_COUNT;

            // Dominant colors (top 3 from global histogram)
            int[] globalColorHist = new int[COLOR_HISTOGRAM_BINS * 3];
            foreach (var cell in cells)
            {
                for (int i = 0; i < globalColorHist.Length; i++)
                    globalColorHist[i] += cell.colorHistogramRaw[i];
            }
            int[][] dominantColors = FindDominantColors(globalColorHist, 3);

            // Build JSON
            float timestamp = (float)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;

            _jsonBuilder.Clear();
            _jsonBuilder.Append("{\"type\":\"summary\"");
            _jsonBuilder.Append(",\"frame_id\":").Append(frameId);
            _jsonBuilder.Append(",\"timestamp\":").Append(timestamp.ToString("F3"));

            // Cells array
            _jsonBuilder.Append(",\"cells\":[");
            for (int i = 0; i < GRID_CELL_COUNT; i++)
            {
                if (i > 0) _jsonBuilder.Append(",");
                AppendCellJson(_jsonBuilder, cells[i]);
            }
            _jsonBuilder.Append("]");

            // Global stats
            _jsonBuilder.Append(",\"global_brightness\":").Append(globalBrightness.ToString("F4"));
            _jsonBuilder.Append(",\"global_edge_density\":").Append(globalEdgeDensity.ToString("F4"));
            _jsonBuilder.Append(",\"global_motion\":").Append(globalMotion.ToString("F4"));

            // Dominant colors
            _jsonBuilder.Append(",\"dominant_colors\":[");
            for (int i = 0; i < dominantColors.Length; i++)
            {
                if (i > 0) _jsonBuilder.Append(",");
                _jsonBuilder.Append("[")
                    .Append(dominantColors[i][0]).Append(",")
                    .Append(dominantColors[i][1]).Append(",")
                    .Append(dominantColors[i][2]).Append("]");
            }
            _jsonBuilder.Append("]");

            _jsonBuilder.Append(",\"high_res_available\":false");
            _jsonBuilder.Append("}");

            string json = _jsonBuilder.ToString();

            if (debugLogging)
            {
                Debug.Log($"[VisualFeatureExtractor] Frame {frameId}: " +
                    $"brightness={globalBrightness:F2} edges={globalEdgeDensity:F2} " +
                    $"motion={globalMotion:F4}");
            }

            return json;
        }

        // =====================================================================
        // Per-Cell Feature Extraction
        // =====================================================================

        private struct CellFeatures
        {
            public int row;
            public int col;
            public float[] colorHistogram;    // Normalized (24 bins)
            public int[] colorHistogramRaw;   // Raw counts (24 bins)
            public float[] edgeDirections;    // Normalized (8 bins)
            public float meanBrightness;
            public float edgeDensity;
            public float motionMagnitude;     // Set after all cells extracted
        }

        private CellFeatures ExtractCellFeatures(int row, int col, int cellW, int cellH)
        {
            // Cell pixel bounds (top-left origin in texture space)
            // Unity textures are bottom-up, so flip row
            int startX = col * cellW;
            int startY = (GRID_ROWS - 1 - row) * cellH;

            // Color histogram: 8 bins per channel × 3 channels = 24
            int[] colorHist = new int[COLOR_HISTOGRAM_BINS * 3];
            float brightnessSum = 0f;

            // Edge detection (simple Sobel-like gradient magnitude)
            int edgePixels = 0;
            int totalPixels = cellW * cellH;
            int[] edgeDirHist = new int[EDGE_DIRECTION_BINS];

            for (int y = 0; y < cellH; y++)
            {
                for (int x = 0; x < cellW; x++)
                {
                    int px = startX + x;
                    int py = startY + y;

                    if (px >= captureWidth || py >= captureHeight) continue;

                    Color32 c = _pixels[py * captureWidth + px];

                    // Color histogram binning
                    int rBin = Mathf.Min(c.r * COLOR_HISTOGRAM_BINS / 256, COLOR_HISTOGRAM_BINS - 1);
                    int gBin = Mathf.Min(c.g * COLOR_HISTOGRAM_BINS / 256, COLOR_HISTOGRAM_BINS - 1);
                    int bBin = Mathf.Min(c.b * COLOR_HISTOGRAM_BINS / 256, COLOR_HISTOGRAM_BINS - 1);
                    colorHist[rBin]++;
                    colorHist[COLOR_HISTOGRAM_BINS + gBin]++;
                    colorHist[COLOR_HISTOGRAM_BINS * 2 + bBin]++;

                    // Brightness (luminance)
                    float brightness = (c.r * 0.299f + c.g * 0.587f + c.b * 0.114f) / 255f;
                    brightnessSum += brightness;

                    // Simple gradient for edge detection (skip borders)
                    if (x > 0 && x < cellW - 1 && y > 0 && y < cellH - 1)
                    {
                        Color32 left = _pixels[py * captureWidth + (px - 1)];
                        Color32 right = _pixels[py * captureWidth + (px + 1)];
                        Color32 above = _pixels[(py + 1) * captureWidth + px];
                        Color32 below = _pixels[(py - 1) * captureWidth + px];

                        float gx = ((right.r + right.g + right.b) - (left.r + left.g + left.b)) / 765f;
                        float gy = ((above.r + above.g + above.b) - (below.r + below.g + below.b)) / 765f;

                        float gradMag = Mathf.Sqrt(gx * gx + gy * gy);

                        if (gradMag > 0.05f)  // Edge threshold
                        {
                            edgePixels++;

                            // Direction bin (0-7 for 8 directions)
                            float angle = Mathf.Atan2(gy, gx) * Mathf.Rad2Deg;
                            if (angle < 0) angle += 360f;
                            int dirBin = Mathf.Min((int)(angle / 45f), EDGE_DIRECTION_BINS - 1);
                            edgeDirHist[dirBin]++;
                        }
                    }
                }
            }

            // Normalize color histogram
            float[] colorHistNorm = new float[COLOR_HISTOGRAM_BINS * 3];
            for (int i = 0; i < colorHistNorm.Length; i++)
                colorHistNorm[i] = (float)colorHist[i] / totalPixels;

            // Normalize edge direction histogram
            float[] edgeDirNorm = new float[EDGE_DIRECTION_BINS];
            int edgeTotal = Mathf.Max(edgePixels, 1);
            for (int i = 0; i < EDGE_DIRECTION_BINS; i++)
                edgeDirNorm[i] = (float)edgeDirHist[i] / edgeTotal;

            return new CellFeatures
            {
                row = row,
                col = col,
                colorHistogram = colorHistNorm,
                colorHistogramRaw = colorHist,
                edgeDirections = edgeDirNorm,
                meanBrightness = brightnessSum / totalPixels,
                edgeDensity = (float)edgePixels / Mathf.Max(totalPixels - (2 * (cellW + cellH) - 4), 1),
                motionMagnitude = 0f  // Set later
            };
        }

        // =====================================================================
        // Dominant Color Extraction
        // =====================================================================

        private int[][] FindDominantColors(int[] globalHist, int count)
        {
            // Find top N bins by frequency across R, G, B channels
            // Each dominant color = (R_bin_center, G_bin_center, B_bin_center)
            int[] rHist = new int[COLOR_HISTOGRAM_BINS];
            int[] gHist = new int[COLOR_HISTOGRAM_BINS];
            int[] bHist = new int[COLOR_HISTOGRAM_BINS];

            for (int i = 0; i < COLOR_HISTOGRAM_BINS; i++)
            {
                rHist[i] = globalHist[i];
                gHist[i] = globalHist[COLOR_HISTOGRAM_BINS + i];
                bHist[i] = globalHist[COLOR_HISTOGRAM_BINS * 2 + i];
            }

            // Find top R, G, B bins independently, then combine
            int[][] result = new int[count][];
            for (int n = 0; n < count; n++)
            {
                int rMax = FindMaxBin(rHist);
                int gMax = FindMaxBin(gHist);
                int bMax = FindMaxBin(bHist);

                // Convert bin index to color midpoint (0-255)
                int binSize = 256 / COLOR_HISTOGRAM_BINS;
                result[n] = new int[]
                {
                    rMax * binSize + binSize / 2,
                    gMax * binSize + binSize / 2,
                    bMax * binSize + binSize / 2
                };

                // Zero out used bins so next iteration finds different color
                if (n < count - 1)
                {
                    rHist[rMax] = 0;
                    gHist[gMax] = 0;
                    bHist[bMax] = 0;
                }
            }

            return result;
        }

        private int FindMaxBin(int[] hist)
        {
            int maxIdx = 0;
            int maxVal = hist[0];
            for (int i = 1; i < hist.Length; i++)
            {
                if (hist[i] > maxVal)
                {
                    maxVal = hist[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // =====================================================================
        // JSON Serialization (manual for performance)
        // =====================================================================

        private void AppendCellJson(StringBuilder sb, CellFeatures cell)
        {
            sb.Append("{\"row\":").Append(cell.row);
            sb.Append(",\"col\":").Append(cell.col);

            // color_histogram
            sb.Append(",\"color_histogram\":[");
            for (int i = 0; i < cell.colorHistogram.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(cell.colorHistogram[i].ToString("F4"));
            }
            sb.Append("]");

            // edge_directions
            sb.Append(",\"edge_directions\":[");
            for (int i = 0; i < cell.edgeDirections.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(cell.edgeDirections[i].ToString("F4"));
            }
            sb.Append("]");

            sb.Append(",\"mean_brightness\":").Append(cell.meanBrightness.ToString("F4"));
            sb.Append(",\"edge_density\":").Append(cell.edgeDensity.ToString("F4"));
            sb.Append(",\"motion_magnitude\":").Append(cell.motionMagnitude.ToString("F4"));
            sb.Append("}");
        }
    }
}

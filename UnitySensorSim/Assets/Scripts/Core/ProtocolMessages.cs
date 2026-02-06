using System;
using System.Collections.Generic;
using UnityEngine;

namespace EpisodicAgent.Protocol
{
    /// <summary>
    /// Protocol version for backward compatibility checking.
    /// </summary>
    public static class ProtocolVersion
    {
        public const string VERSION = "1.0.0";
    }

    // =========================================================================
    // SENSOR FRAME MESSAGES (Unity → Python)
    // =========================================================================

    /// <summary>
    /// Complete sensor frame sent to the Python agent.
    /// 
    /// ARCHITECTURAL INVARIANT: Unity does not include semantic labels.
    /// All labels are owned by the Python backend and learned from user interaction.
    /// This frame provides only observable properties: GUIDs, positions, states.
    /// </summary>
    [Serializable]
    public class SensorFrame
    {
        public string protocol_version;
        public string timestamp;       // ISO 8601 format
        public int frame_id;
        public CameraPose camera;
        public string current_room_guid;
        // REMOVED: current_room_label - backend owns all semantic labeling
        public List<EntityData> entities;
        public List<StateChangeEvent> state_changes;

        public SensorFrame()
        {
            protocol_version = ProtocolVersion.VERSION;
            entities = new List<EntityData>();
            state_changes = new List<StateChangeEvent>();
        }
    }

    /// <summary>
    /// Camera/player pose in world space.
    /// </summary>
    [Serializable]
    public class CameraPose
    {
        public Vector3Data position;
        public Vector3Data forward;
        public Vector3Data up;
        public float yaw;    // Horizontal rotation in degrees
        public float pitch;  // Vertical rotation in degrees
    }

    /// <summary>
    /// Serializable Vector3 for JSON.
    /// </summary>
    [Serializable]
    public class Vector3Data
    {
        public float x;
        public float y;
        public float z;

        public Vector3Data() { }

        public Vector3Data(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }

        public Vector3 ToVector3() => new Vector3(x, y, z);
    }

    /// <summary>
    /// Entity data included in each frame.
    /// 
    /// ARCHITECTURAL INVARIANT: Unity does not include semantic labels.
    /// Only observable properties are sent: GUID, position, size, state, visibility.
    /// </summary>
    [Serializable]
    public class EntityData
    {
        public string guid;
        // REMOVED: label and category - backend owns all semantic labeling
        public Vector3Data position;
        public Vector3Data size;           // Approximate bounding box size
        public float distance;             // Distance from camera
        public bool is_visible;            // Currently visible to player
        public string interactable_type;   // null, "toggle", "moveable"
        public string interactable_state;  // e.g., "open", "closed", "on", "off"
    }

    /// <summary>
    /// State change notification (optional, for events between frames).
    /// </summary>
    [Serializable]
    public class StateChangeEvent
    {
        public string entity_guid;
        public string change_type;    // "state_changed", "spawned", "despawned", "moved"
        public string old_value;
        public string new_value;
        public string timestamp;
    }

    // =========================================================================
    // COMMAND MESSAGES (Python → Unity)
    // =========================================================================

    /// <summary>
    /// Base command structure from Python.
    /// </summary>
    [Serializable]
    public class CommandMessage
    {
        public string protocol_version;
        public string command_id;      // Unique ID for response correlation
        public string command;         // Command type
        public CommandParams parameters;
    }

    /// <summary>
    /// Command parameters (flexible for different commands).
    /// </summary>
    [Serializable]
    public class CommandParams
    {
        // For teleport
        public string room_guid;

        // For toggle interactable
        public string entity_guid;
        public string target_state;  // Optional: force specific state

        // For spawn/move ball
        public Vector3Data position;

        // For generic entity operations
        public string entity_label;

        // ---- Dynamic visualization commands (backend → Unity) ----

        // For create_room_volume / update_room_volume
        public string location_id;       // Backend-assigned location ID
        public string label;             // Label for display
        public Vector3Data center;       // Center position of volume
        public Vector3Data extent;       // Half-extents (box size / 2)
        public float radius;             // Radius for sphere volumes
        public string color;             // Hex color string (e.g., "#FF8800")
        public float opacity;            // 0.0–1.0 for semi-transparent volumes

        // For set_entity_label
        // entity_guid + label (reused from above)

        // For clear_dynamic_volumes
        // No extra params needed
    }

    /// <summary>
    /// Command response sent back to Python.
    /// </summary>
    [Serializable]
    public class CommandResponse
    {
        public string protocol_version;
        public string command_id;
        public bool success;
        public string error_message;
        public string timestamp;

        public CommandResponse()
        {
            protocol_version = ProtocolVersion.VERSION;
        }

        public static CommandResponse Success(string commandId)
        {
            return new CommandResponse
            {
                command_id = commandId,
                success = true,
                timestamp = DateTime.UtcNow.ToString("o")
            };
        }

        public static CommandResponse Error(string commandId, string message)
        {
            return new CommandResponse
            {
                command_id = commandId,
                success = false,
                error_message = message,
                timestamp = DateTime.UtcNow.ToString("o")
            };
        }
    }

    // =========================================================================
    // SUPPORTED COMMANDS
    // =========================================================================

    /// <summary>
    /// Enum of supported command types.
    /// </summary>
    public static class CommandTypes
    {
        public const string TELEPORT_PLAYER = "teleport_player";
        public const string TOGGLE_INTERACTABLE = "toggle_interactable";
        public const string SPAWN_BALL = "spawn_ball";
        public const string DESPAWN_BALL = "despawn_ball";
        public const string MOVE_BALL = "move_ball";
        public const string RESET_WORLD = "reset_world";
        public const string GET_WORLD_STATE = "get_world_state";

        // Dynamic visualization commands (Python backend → Unity overlay)
        public const string CREATE_ROOM_VOLUME = "create_room_volume";
        public const string UPDATE_ROOM_VOLUME = "update_room_volume";
        public const string REMOVE_ROOM_VOLUME = "remove_room_volume";
        public const string SET_ENTITY_LABEL = "set_entity_label";
        public const string CLEAR_DYNAMIC_VOLUMES = "clear_dynamic_volumes";
    }
}

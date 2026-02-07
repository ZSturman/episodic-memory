using System;
using UnityEngine;
using EpisodicAgent.Protocol;
using EpisodicAgent.World;

namespace EpisodicAgent.Core
{
    /// <summary>
    /// Receives and processes commands from Python clients.
    /// Supports teleportation, interactable toggling, ball manipulation,
    /// world reset, and dynamic visualization overlays.
    /// </summary>
    public class CommandReceiver : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private WebSocketServer webSocketServer;
        [SerializeField] private WorldManager worldManager;
        [SerializeField] private DynamicWorldBuilder dynamicWorldBuilder;
        [SerializeField] private Transform player;
        [SerializeField] private CharacterController playerController;  // For teleportation

        // Events for external handling
        public event Action<string, bool, string> OnCommandProcessed;  // command_id, success, error
        /// <summary>Fires with the raw JSON string of every incoming command. Useful for debugging.</summary>
        public event Action<string> OnCommandJsonReceived;
        /// <summary>Fires with the raw JSON string of every outgoing response. Useful for debugging.</summary>
        public event Action<string> OnResponseJsonSent;

        private void Start()
        {
            if (webSocketServer == null)
            {
                webSocketServer = GetComponent<WebSocketServer>();
            }

            if (webSocketServer != null)
            {
                webSocketServer.OnMessageReceived += HandleMessage;
            }

            if (worldManager == null)
            {
                worldManager = FindFirstObjectByType<WorldManager>();
            }

            if (dynamicWorldBuilder == null)
            {
                dynamicWorldBuilder = FindFirstObjectByType<DynamicWorldBuilder>();
            }
        }

        private void OnDestroy()
        {
            if (webSocketServer != null)
            {
                webSocketServer.OnMessageReceived -= HandleMessage;
            }
        }

        private void HandleMessage(string json)
        {
            CommandMessage command;
            try
            {
                command = JsonUtility.FromJson<CommandMessage>(json);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[CommandReceiver] Failed to parse command: {ex.Message}");
                return;
            }

            if (command == null || string.IsNullOrEmpty(command.command))
            {
                Debug.LogWarning("[CommandReceiver] Received invalid command");
                return;
            }

            Debug.Log($"[CommandReceiver] Processing command: {command.command} (id: {command.command_id})");

            OnCommandJsonReceived?.Invoke(json);

            CommandResponse response = ProcessCommand(command);
            SendResponse(response);

            OnCommandProcessed?.Invoke(command.command_id, response.success, response.error_message);
        }

        private CommandResponse ProcessCommand(CommandMessage command)
        {
            try
            {
                switch (command.command)
                {
                    case CommandTypes.TELEPORT_PLAYER:
                        return HandleTeleportPlayer(command);

                    case CommandTypes.TOGGLE_INTERACTABLE:
                        return HandleToggleInteractable(command);

                    case CommandTypes.SPAWN_BALL:
                        return HandleSpawnBall(command);

                    case CommandTypes.DESPAWN_BALL:
                        return HandleDespawnBall(command);

                    case CommandTypes.MOVE_BALL:
                        return HandleMoveBall(command);

                    case CommandTypes.RESET_WORLD:
                        return HandleResetWorld(command);

                    case CommandTypes.GET_WORLD_STATE:
                        return HandleGetWorldState(command);

                    // Dynamic visualization commands
                    case CommandTypes.CREATE_ROOM_VOLUME:
                        return HandleCreateRoomVolume(command);

                    case CommandTypes.UPDATE_ROOM_VOLUME:
                        return HandleUpdateRoomVolume(command);

                    case CommandTypes.REMOVE_ROOM_VOLUME:
                        return HandleRemoveRoomVolume(command);

                    case CommandTypes.SET_ENTITY_LABEL:
                        return HandleSetEntityLabel(command);

                    case CommandTypes.CLEAR_DYNAMIC_VOLUMES:
                        return HandleClearDynamicVolumes(command);

                    default:
                        return CommandResponse.Error(command.command_id, 
                            $"Unknown command: {command.command}");
                }
            }
            catch (Exception ex)
            {
                return CommandResponse.Error(command.command_id, ex.Message);
            }
        }

        private CommandResponse HandleTeleportPlayer(CommandMessage command)
        {
            if (command.parameters == null || string.IsNullOrEmpty(command.parameters.room_guid))
            {
                return CommandResponse.Error(command.command_id, "Missing room_guid parameter");
            }

            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            RoomVolume room = worldManager.GetRoomByGuid(command.parameters.room_guid);
            if (room == null)
            {
                return CommandResponse.Error(command.command_id, 
                    $"Room not found: {command.parameters.room_guid}");
            }

            // Get spawn point in room
            Vector3 spawnPoint = room.GetSpawnPoint();

            // Teleport player
            if (playerController != null)
            {
                playerController.enabled = false;
                player.position = spawnPoint;
                playerController.enabled = true;
            }
            else if (player != null)
            {
                player.position = spawnPoint;
            }
            else
            {
                return CommandResponse.Error(command.command_id, "Player reference not set");
            }

            Debug.Log($"[CommandReceiver] Teleported player to room: {room.Label}");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleToggleInteractable(CommandMessage command)
        {
            if (command.parameters == null || string.IsNullOrEmpty(command.parameters.entity_guid))
            {
                return CommandResponse.Error(command.command_id, "Missing entity_guid parameter");
            }

            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            EntityMarker entity = worldManager.GetEntityByGuid(command.parameters.entity_guid);
            if (entity == null)
            {
                return CommandResponse.Error(command.command_id, 
                    $"Entity not found: {command.parameters.entity_guid}");
            }

            InteractableState interactable = entity.GetComponent<InteractableState>();
            if (interactable == null)
            {
                return CommandResponse.Error(command.command_id, 
                    $"Entity is not interactable: {entity.Label}");
            }

            // Toggle or set specific state
            if (!string.IsNullOrEmpty(command.parameters.target_state))
            {
                interactable.SetState(command.parameters.target_state);
            }
            else
            {
                interactable.Toggle();
            }

            Debug.Log($"[CommandReceiver] Toggled interactable: {entity.Label} -> {interactable.CurrentState}");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleSpawnBall(CommandMessage command)
        {
            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            Vector3 position = Vector3.zero;
            if (command.parameters?.position != null)
            {
                position = command.parameters.position.ToVector3();
            }
            else if (player != null)
            {
                // Spawn in front of player
                position = player.position + player.forward * 2f;
            }

            bool success = worldManager.SpawnBall(position);
            if (!success)
            {
                return CommandResponse.Error(command.command_id, "Failed to spawn ball");
            }

            Debug.Log($"[CommandReceiver] Spawned ball at {position}");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleDespawnBall(CommandMessage command)
        {
            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            bool success = worldManager.DespawnBall();
            if (!success)
            {
                return CommandResponse.Error(command.command_id, "No ball to despawn");
            }

            Debug.Log("[CommandReceiver] Despawned ball");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleMoveBall(CommandMessage command)
        {
            if (command.parameters?.position == null)
            {
                return CommandResponse.Error(command.command_id, "Missing position parameter");
            }

            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            Vector3 position = command.parameters.position.ToVector3();
            bool success = worldManager.MoveBall(position);
            if (!success)
            {
                return CommandResponse.Error(command.command_id, "No ball to move (spawn first)");
            }

            Debug.Log($"[CommandReceiver] Moved ball to {position}");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleResetWorld(CommandMessage command)
        {
            if (worldManager == null)
            {
                return CommandResponse.Error(command.command_id, "WorldManager not available");
            }

            worldManager.ResetWorld();
            Debug.Log("[CommandReceiver] World reset");
            return CommandResponse.Success(command.command_id);
        }

        private CommandResponse HandleGetWorldState(CommandMessage command)
        {
            // This could return a full world state snapshot
            // For now, just acknowledge
            Debug.Log("[CommandReceiver] World state requested (not fully implemented)");
            return CommandResponse.Success(command.command_id);
        }

        private void SendResponse(CommandResponse response)
        {
            if (webSocketServer == null) return;

            string json = JsonUtility.ToJson(response);

            OnResponseJsonSent?.Invoke(json);

            webSocketServer.Broadcast(json);
        }

        // =================================================================
        // Dynamic visualization command handlers
        // =================================================================

        private CommandResponse HandleCreateRoomVolume(CommandMessage command)
        {
            if (dynamicWorldBuilder == null)
                return CommandResponse.Error(command.command_id, "DynamicWorldBuilder not available");

            var p = command.parameters;
            if (p == null || string.IsNullOrEmpty(p.location_id))
                return CommandResponse.Error(command.command_id, "Missing location_id");

            Vector3 center = p.center != null ? p.center.ToVector3() : Vector3.zero;
            Vector3 extent = p.extent != null ? p.extent.ToVector3() : Vector3.one;
            string label = p.label ?? p.location_id;

            Color? color = null;
            if (!string.IsNullOrEmpty(p.color))
                color = DynamicWorldBuilder.ParseHexColor(p.color, Color.cyan);

            float opacity = p.opacity > 0 ? p.opacity : 0.15f;

            bool ok = dynamicWorldBuilder.CreateVolume(p.location_id, label, center, extent, color, opacity);
            return ok
                ? CommandResponse.Success(command.command_id)
                : CommandResponse.Error(command.command_id, "Failed to create volume");
        }

        private CommandResponse HandleUpdateRoomVolume(CommandMessage command)
        {
            if (dynamicWorldBuilder == null)
                return CommandResponse.Error(command.command_id, "DynamicWorldBuilder not available");

            var p = command.parameters;
            if (p == null || string.IsNullOrEmpty(p.location_id))
                return CommandResponse.Error(command.command_id, "Missing location_id");

            Vector3? center = p.center != null ? p.center.ToVector3() : null;
            Vector3? extent = p.extent != null ? p.extent.ToVector3() : null;
            Color? color = !string.IsNullOrEmpty(p.color)
                ? DynamicWorldBuilder.ParseHexColor(p.color, Color.cyan)
                : null;
            float? opacity = p.opacity > 0 ? p.opacity : null;

            bool ok = dynamicWorldBuilder.UpdateVolume(p.location_id, p.label, center, extent, color, opacity);
            return ok
                ? CommandResponse.Success(command.command_id)
                : CommandResponse.Error(command.command_id, $"Volume not found: {p.location_id}");
        }

        private CommandResponse HandleRemoveRoomVolume(CommandMessage command)
        {
            if (dynamicWorldBuilder == null)
                return CommandResponse.Error(command.command_id, "DynamicWorldBuilder not available");

            var p = command.parameters;
            if (p == null || string.IsNullOrEmpty(p.location_id))
                return CommandResponse.Error(command.command_id, "Missing location_id");

            bool ok = dynamicWorldBuilder.RemoveVolume(p.location_id);
            return ok
                ? CommandResponse.Success(command.command_id)
                : CommandResponse.Error(command.command_id, $"Volume not found: {p.location_id}");
        }

        private CommandResponse HandleSetEntityLabel(CommandMessage command)
        {
            if (dynamicWorldBuilder == null)
                return CommandResponse.Error(command.command_id, "DynamicWorldBuilder not available");

            var p = command.parameters;
            if (p == null || string.IsNullOrEmpty(p.entity_guid) || string.IsNullOrEmpty(p.label))
                return CommandResponse.Error(command.command_id, "Missing entity_guid or label");

            bool ok = dynamicWorldBuilder.SetEntityLabel(p.entity_guid, p.label);
            return ok
                ? CommandResponse.Success(command.command_id)
                : CommandResponse.Error(command.command_id, $"Entity not found: {p.entity_guid}");
        }

        private CommandResponse HandleClearDynamicVolumes(CommandMessage command)
        {
            if (dynamicWorldBuilder == null)
                return CommandResponse.Error(command.command_id, "DynamicWorldBuilder not available");

            dynamicWorldBuilder.ClearAllVolumes();
            return CommandResponse.Success(command.command_id);
        }
    }
}

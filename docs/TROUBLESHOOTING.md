# Troubleshooting Guide

Common issues and solutions for the Episodic Memory Agent.

## Connection Issues

### Problem: "Connection refused" or WebSocket error

**Symptoms:**
```
Connection failed: [Errno 111] Connection refused
WebSocket connection refused
```

**Solutions:**

1. **Ensure Unity is in Play mode**
   - Click the Play button (▶) in Unity
   - Check Console shows "WebSocket server started"

2. **Verify port is not in use**
   ```bash
   # Check what's using port 8765
   lsof -i :8765
   
   # Kill process if needed
   kill -9 <PID>
   ```

3. **Try explicit localhost**
   ```bash
   python -m episodic_agent run --unity-ws ws://127.0.0.1:8765
   ```

4. **Check firewall settings**
   - Allow localhost connections
   - Temporarily disable firewall for testing

---

### Problem: "Invalid frame format" or Schema validation failed

**Symptoms:**
```
Schema validation failed
Invalid frame format: missing field 'frame_id'
```

**Solutions:**

1. **Update protocol schemas**
   - Check `UnitySensorSim/protocol/sensor_frame_schema.json`
   - Ensure Unity and Python use matching versions

2. **Verify Unity scripts**
   - Check `SensorStreamer.cs` for correct JSON format
   - Look for compile errors in Unity Console

3. **Enable verbose logging**
   ```bash
   python -m episodic_agent run --verbose
   ```

---

### Problem: No frames received (agent appears stuck)

**Symptoms:**
- Agent starts but never shows step updates
- "Waiting for frame..." message

**Solutions:**

1. **Move in Unity**
   - Frames may only send on movement
   - Walk around to trigger updates

2. **Check SensorStreamer settings**
   - Ensure `Target Frame Rate` > 0
   - Verify `SensorStreamer` is enabled

3. **Check WorldManager**
   - Ensure Player reference is set
   - Look for null reference errors

4. **Lower FPS**
   ```bash
   python -m episodic_agent run --fps 5
   ```

---

## Location Issues

### Problem: Location always shows "unknown"

**Symptoms:**
```
[0001] 📍 unknown(0.00) 👁 [] 📚 0
```

**Solutions:**

1. **Verify room setup**
   - Room has `RoomVolume` component
   - Box Collider has `Is Trigger` enabled
   - Collider is large enough to contain player

2. **Check player position**
   - Player must be inside room trigger
   - Use Unity's Scene view to verify

3. **Check console for errors**
   - Look for null reference exceptions
   - Verify GUID is being set

---

### Problem: Re-entering room prompts for label again

**Symptoms:**
- Previously labeled room asks for label again
- Location not persisting between runs

**Solutions:**

1. **Check persistence files**
   ```bash
   ls runs/<timestamp>/
   # Should have nodes.jsonl
   ```

2. **Verify GUID stability**
   - Room GUID should be same each session
   - Check `RoomVolume` component in Unity

3. **Use auto-labeling for testing**
   ```bash
   python -m episodic_agent run --auto-label
   ```

---

## Event Detection Issues

### Problem: State changes not detected

**Symptoms:**
- Toggle door in Unity but no event shown
- `state_changes` array always empty

**Solutions:**

1. **Verify InteractableState**
   - Entity has `InteractableState` component
   - State Type is set (OnOff or OpenClosed)

2. **Check entity is visible**
   - Entity must be in camera view
   - Distance not too far

3. **Toggle correctly**
   - Press E to interact in Unity
   - Check Unity Console for toggle messages

---

### Problem: Too many events detected

**Symptoms:**
- Same event detected multiple times
- Event spam in logs

**Solutions:**

1. **Check frame rate**
   - Lower FPS reduces duplicate events
   ```bash
   python -m episodic_agent run --fps 5
   ```

2. **Verify delta detector settings**
   - Check `missing_window` parameter
   - Increase threshold values

---

## Episode Issues

### Problem: Episodes never freeze

**Symptoms:**
- Episode count stays at 0
- No `📦` marker in output

**Solutions:**

1. **Check boundary settings**
   - Default may require location change
   - Try walking to different room

2. **Verify time-based boundary**
   - Check `--freeze-interval` setting
   ```bash
   python -m episodic_agent run --freeze-interval 50
   ```

3. **Use stub profile for testing**
   ```bash
   python -m episodic_agent run --profile stub --freeze-interval 30
   ```

---

### Problem: Episodes freeze too often

**Symptoms:**
- New episode every few steps
- Boundary reasons: "location_flicker"

**Solutions:**

1. **Adjust hysteresis thresholds**
   - Increase boundary thresholds in profile
   - Use `unity_full` profile (has hysteresis)

2. **Stabilize room detection**
   - Make room colliders larger
   - Reduce overlap between rooms

---

## Scenario Issues

### Problem: Scenario fails with "Command timeout"

**Symptoms:**
```
Command timeout: cmd-001
Scenario failed: toggle_drawer_light
```

**Solutions:**

1. **Ensure Unity is responsive**
   - Not paused or minimized
   - No dialog boxes open

2. **Check entity GUIDs**
   - Scenario uses hardcoded GUIDs
   - Verify GUIDs match scene

3. **Try offline replay**
   ```bash
   python -m episodic_agent scenario walk_rooms --replay replay.jsonl
   ```

---

### Problem: Scenario command not recognized

**Symptoms:**
```
Unknown command: teleport_player
Command not implemented
```

**Solutions:**

1. **Update Unity scripts**
   - Check `CommandReceiver.cs` has command
   - Recompile Unity scripts

2. **Check protocol version**
   - Ensure Python and Unity use same version

---

## Report Issues

### Problem: Report generation fails

**Symptoms:**
```
Error generating report: KeyError 'steps'
FileNotFoundError: run.jsonl
```

**Solutions:**

1. **Verify run completed**
   - Check run folder has required files
   ```bash
   ls runs/<timestamp>/
   # Should have: run.jsonl, episodes.jsonl
   ```

2. **Check file permissions**
   ```bash
   chmod 644 runs/<timestamp>/*.jsonl
   ```

3. **Compute metrics first**
   ```bash
   python -m episodic_agent report runs/<timestamp> --compute-metrics
   ```

---

### Problem: Empty report or all zeros

**Symptoms:**
- Report shows 0 steps, 0 episodes
- No data in visualizations

**Solutions:**

1. **Run longer**
   - Need sufficient steps for meaningful data
   ```bash
   python -m episodic_agent run --steps 200
   ```

2. **Check log format**
   - Ensure `run.jsonl` has valid JSON lines
   ```bash
   head -5 runs/<timestamp>/run.jsonl | jq .
   ```

---

## Performance Issues

### Problem: High CPU usage

**Solutions:**

1. **Lower frame rate**
   ```bash
   python -m episodic_agent run --fps 5
   ```

2. **Disable verbose logging**
   - Remove `--verbose` flag

3. **Use stub profile for testing**

---

### Problem: Memory usage grows over time

**Solutions:**

1. **Enable episode freezing**
   - Clears working memory periodically

2. **Limit step count**
   ```bash
   python -m episodic_agent run --steps 1000
   ```

---

## Getting Help

If none of these solutions work:

1. **Check logs**
   ```bash
   cat runs/<timestamp>/run.jsonl | tail -20
   ```

2. **Enable debug mode**
   ```bash
   python -m episodic_agent run --verbose 2>&1 | tee debug.log
   ```

3. **Check Unity Console**
   - Red errors indicate problems
   - Copy full error message

4. **Verify installation**
   ```bash
   pip install -e ".[dev]" --force-reinstall
   ```

from time import sleep
import uiautomator2 as ui
from collections import deque, defaultdict
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import re
import subprocess
import threading
import queue
import json
# (web UI removed — keep imports minimal)
command_queue = queue.Queue()
import paho.mqtt.client as mqtt
from sec import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USER,
    MQTT_PASSWORD,
    HA_DISCOVERY_PREFIX,
DEVICE_NAME,
)

d = ui.connect_usb()
#connect("192.168.178.64:5555") #_usb()
pw = "1909"

def queue_task(func, *args, **kwargs):
    """Enqueue a callable task to be processed sequentially by the command worker."""
    def _task():
        func(*args, **kwargs)
    command_queue.put(_task)

# --------------------------
# Cached XML Helper
# --------------------------
TREE_CACHE = None

def refresh_tree():
    print("Refresh the global UI tree.")
    global TREE_CACHE
    xml_str = d.dump_hierarchy()
    with open("ui_dump.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)
    TREE_CACHE = ET.fromstring(xml_str)

def get_tree():
    """Get cached tree; refresh if None."""
    global TREE_CACHE
    if TREE_CACHE is None:
        refresh_tree()
    return TREE_CACHE

def clear_tree():
    print("Invalidate the global UI tree cache.")
    """Invalidate cache after interaction."""
    global TREE_CACHE
    TREE_CACHE = None

# --------------------------
# XML Query Helpers
# --------------------------
# Step 1: find the parent node with the desired resource-id
def find_by_text(root, text, contains=False):
    for elem in root.iter('node'):
        t = elem.attrib.get('text','')
        if (contains and text in t) or (not contains and t == text):
            return elem
    return None

def find_by_desc(root, desc, contains=False):
    for elem in root.iter('node'):
        dsc = elem.attrib.get('content-desc','')
        if (contains and desc in dsc) or (not contains and dsc == desc):
            return elem
    return None

def getRobotStatusBar():
    """
    Locate a structure of three nested android.view.View (index == 0) elements
    followed by a child android.widget.TextView with index 0. Returns the TextView node or None.
    """

    def _first_child(node, cls):
        for child in list(node):
            if child.attrib.get("class") == cls and child.attrib.get("index") == "0":
                return child
        return None

    for elem in get_tree().iter("node"):
        if elem.attrib.get("class") != "android.view.View" or elem.attrib.get("index") != "0":
            continue
        level1 = _first_child(elem, "android.view.View")
        if level1 is None:
            continue
        level2 = _first_child(level1, "android.view.View")
        if level2 is None:
            continue
        target = _first_child(level2, "android.widget.TextView")
        if target is not None:
            return target.attrib.get("text", None)
    return None

def click_elem(elem):
    """Click element using bounds and invalidate tree cache."""
    if elem is None:
        return False
    bounds = elem.attrib['bounds']  # e.g., "[0,1443][1080,1600]"
    # Extract four numbers from the string
    numbers = list(map(int, re.findall(r'\d+', bounds)))
    if len(numbers) != 4:
        print("Warning: invalid bounds:", bounds)
        return False
    x1, y1, x2, y2 = numbers
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    d.click(x, y)
    clear_tree()
    return True

def swipe(*args, **kwargs):
    """Swipe wrapper that invalidates tree."""
    d.swipe(*args, **kwargs)
    clear_tree()

def drag(*args, **kwargs):
    """Drag wrapper that invalidates tree."""
    d.drag(*args, **kwargs)
    clear_tree()

# --------------------------
# Page Detection
# --------------------------
def inScreenOff(): return not d.info.get('screenOn')
def inLock(): return find_by_desc(get_tree(), "Entsperren") is not None
def inDesktop(): return find_by_desc(get_tree(), "Nova-Suche") is not None
def inMain(): return find_by_desc(get_tree(), "Enter") is not None
def inScenario(): return find_by_text(get_tree(), "Scenario Clean") is not None and find_by_text(get_tree(), "Nora") is not None
def inRobot():    return find_by_text(get_tree(), "Corridor", contains=True) is not None and find_by_text(get_tree(), "Suction Power") is None and (
            find_by_text(get_tree(), "Start") is not None or find_by_text(get_tree(), "Pause") is not None or find_by_text(get_tree(), "End") is not None)
def inRobotSettings():    return find_by_text(get_tree(), "Corridor", contains=True) is not None and find_by_text(get_tree(), "Suction Power") is not None and (
            find_by_text(get_tree(), "Start") is not None or find_by_text(get_tree(), "Pause") is not None or find_by_text(get_tree(), "End") is not None)
def inStation():    return find_by_text(get_tree(), "Corridor", contains=True) is not None and not inRobot() and not inRobotSettings()
def inStationAdvanced():
    for t in ["Mop Wash Settings", "Auto-Empty settings", "Hot Air Drying Settings"]:
        if find_by_text(get_tree(), t) is not None: return True
    return False
def inWarning(): return find_by_text(get_tree(), "Ignore") is not None and find_by_text(get_tree(), "View") is not None

# --------------------------
# Navigation
# --------------------------
def None2Desktop(): d.press("home"); clear_tree()
def ScreenOff2Lock(): d.screen_on(); clear_tree()
def Lock2Desktop():
    swipe(0.5, 0.8, 0.5, 0.5, 0.1)
    for ch in pw:
        click_elem(find_by_text(get_tree(), ch)); 
def Desktop2Main(): click_elem(find_by_text(get_tree(), "ECOVACS HOME", contains=True))
def Main2Scenario(): click_elem(find_by_desc(get_tree(), "Scenario Clean"))
def Scenario2Main(): d.click(0.5,0.5); clear_tree()
def Main2Robot(): click_elem(find_by_desc(get_tree(), "Enter"))
def RobotSettings2Desktop(): d.press("back"); clear_tree()
def Robot2Main(): click_elem(find_by_text(get_tree(), "Back"))
def Station2Main(): click_elem(find_by_text(get_tree(), "Back"))
def Robot2Station(): click_elem(find_by_text(get_tree(), "Station"))
def Station2Robot(): click_elem(find_by_text(get_tree(), "DEEBOT ", contains=True))
def Warning2None(): click_elem(find_by_text(get_tree(), "Ignore"))

# --------------------------
# NAV_GRAPH
# --------------------------
NAV_GRAPH = defaultdict(dict)
for name, func in list(globals().items()):
    if "2" in name and callable(func):
        src,dst = name.split("2")
        NAV_GRAPH[src][dst] = func

# --------------------------
# Navigation Logic
# --------------------------
def detect_current_page():
    PAGES = {name[2:]: func for name,func in globals().items() if name.startswith("in") and callable(func)}
    for _ in range(10):
        for name, func in PAGES.items():
            if func(): 
                print("Current page:", name)
                return name
        sleep(1)
        print("Retrying page detection...")
        refresh_tree()
    return "None"

def find_path(start, goal, graph):
    queue = deque([(start, [])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current==goal: return path
        visited.add(current)
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                queue.append((neighbor, path+[(current, neighbor)]))
    return None

def navigate_to(target_page):
    for _ in range(10):
        current = detect_current_page()
        if current == target_page: 
            print("already at ", target_page)
            return
        path = find_path(current, target_page, NAV_GRAPH)
        if not path: return
        for src,dst in path:
            print(f"Navigating {src} -> {dst}")
            NAV_GRAPH[src][dst]()
            if detect_current_page() == dst: 
                print(f"Arrived at {dst}")
                break
            break

def ClickPause():
    navigate_to("Robot")
    click_elem(find_by_text(get_tree(), "Pause"))

def ClickEnd():
    navigate_to("Robot")
    click_elem(find_by_text(get_tree(), "End"))

def ClickStart():
    navigate_to("Robot")
    start = find_by_text(get_tree(), "Start")
    if start is None:
        start = find_by_text(get_tree(), "Continue")
    print("Start button:", start.attrib.get('text',''))
    click_elem(start)


# --------------------------
# MapScreenshot
# --------------------------
def MapScreenshot():
    navigate_to("Robot")

    dismissWaringsAndLog()
    centerMap()

    img = d.screenshot()
    w,h = img.size
    img = img.resize((w //  2, h // 2), Image.LANCZOS)
    w,h = img.size
    img = img.crop((0,int(h*0.09),w,int(h*0.55))).convert("RGBA")
    ImageDraw.floodfill(img, xy=(0, -1), value=(255, 255, 255, 0), thresh=25)
    img.save("Map_cropped.png")
    # data = np.array(img)
    # mask = np.all(np.abs(data[:,:,:3].astype(int)-data[0,0,:3].astype(int))<=2, axis=-1)
    # data[mask,:]=[255,255,255,0]
    # Image.fromarray(data).save("Map_cropped.png")
    print("Map screenshot saved.")
    try:
        subprocess.run(["scp", "Map_cropped.png", "hassio:/root/config/www/"], check=True)
        print("File successfully copied to Home Assistant.")
    except subprocess.CalledProcessError as e:
        print("Error during SCP:", e)
    # ----------- update Status ---------------
    # index="1" text="Clean Water Tank short of water or not installed" resource-id="" class="android.widget.TextView"
    status_text = "Unknown"
    status_elem = find_by_desc(get_tree(), "Benachrichtigung von ECOVACS HOME:", contains=True)
    if status_elem is not None:
        status_text = status_elem.attrib.get("content-desc", "")
        status_text = status_text.replace("Benachrichtigung von ECOVACS HOME: ", "")
    else:
        text_elem = find_by_text(get_tree(), "Clean Water Tank short of water or not installed")
        if text_elem is not None:
            status_text = text_elem.attrib.get("text", "")
    status_text = status_text.strip() or getRobotStatusBar()
    print("Status:", status_text)
    global map_status_entity, last_map_status
    last_map_status = status_text
    if map_status_entity is not None:
        map_status_entity.publish_state(status_text)
        print(f"📤 MQTT map status -> {status_text}")
    else:
        print("⚠️ Map status entity not initialized; skipping MQTT publish")
    clear_tree()
    
MAP_REFRESH_INTERVAL_CLEANING = 10
MAP_REFRESH_INTERVAL_IDLE = 3600
map_refresh_timer = None

def map_refresh_task():
    """Run a map screenshot and schedule the next refresh."""
    MapScreenshot()
    schedule_map_refresh()

def schedule_map_refresh():
    """Schedule the next map screenshot based on the current robot status."""
    global map_refresh_timer
    status = (last_map_status or "").strip().lower()
    interval = MAP_REFRESH_INTERVAL_CLEANING if status.startswith("clean") else MAP_REFRESH_INTERVAL_IDLE
    if map_refresh_timer is not None:
        map_refresh_timer.cancel()
    map_refresh_timer = threading.Timer(interval, lambda: queue_task(map_refresh_task))
    map_refresh_timer.daemon = True
    map_refresh_timer.start()
    print(f"🗓️ Next map refresh scheduled in {interval} seconds (status: {last_map_status})")
    
def dismissWaringsAndLog():
    # for elem in get_tree().iter():  # iterate over all elements
    #     if elem.attrib.get("index") == "2" \
    #     and elem.attrib.get("text", "") == "" \
    #     and "Image" in elem.attrib.get("text", ""):  # optional
    #         if elem.attrib.get("desc", "") == "":
    #             click_elem(elem)  # your function that clicks by bounds
    #             break
    cleaningLog = find_by_text(get_tree(), "Cleaning completed. Tap to view the Log.")
    if cleaningLog is not None:
        bounds = cleaningLog.attrib['bounds']
        x1,y1,x2,y2 = map(int, bounds.replace('[','').replace(']',' ').replace(',',' ').split())
        x,y = x2+130,(y1+y2)/2
        d.click(x,y); clear_tree()
        print("Click to dismiss cleaning log at", x,y)

def centerMap():
    corridor = find_by_text(get_tree(), "Corridor", contains=True)
    if corridor is not None:
        bounds = corridor.attrib['bounds']
        x1,y1,x2,y2 = map(int, bounds.replace('[','').replace(']',' ').replace(',',' ').split())
        x,y = (x1+x2)/2,(y1+y2)/2
        print("Corridor bounds:", bounds, x,y)
        if y<630 or y>640 or x<360 or x>400:
            d.double_click(0.5,0.5,0.001)
            d.drag(0.5,0.5,0.5,0.38,0.05)

# --------------------------
# Scenario / Zone Clean
# --------------------------
def ClickZone():
    navigate_to("Robot")
    click_elem(find_by_text(get_tree(), "Zone"))
    zone_elem = find_by_text(get_tree(), "1.0m * 1.0m")
    if zone_elem is not None:
        children = list(zone_elem.iterfind('../*'))
        print(f"Found {len(children)} child elements")
    else: print('Object "1.0m * 1.0m" not found.')

def ClickNora():
    navigate_to("Scenario")
    # clear_tree()
    click_elem(find_by_text(get_tree(), "Nora"))

def ClickPostMeal():
    navigate_to("Scenario")
    clear_tree()
    click_elem(find_by_desc(get_tree(), "Post-meal Clean"))

def ClickStopDryMop():
    navigate_to("Station")
    cancel = find_by_text(get_tree(), "Cancel")
    if cancel is not None:
        bounds = cancel.attrib['bounds']
        x1,y1,x2,y2 = map(int, bounds.replace('[','').replace(']',' ').replace(',',' ').split())
        x,y = (x1+x2)/2,y1-100
        d.click(x,y); clear_tree() 

_MQTT_ENTITY_CONTEXT = {
    "client": None,
    "device_info": None,
    "ha_prefix": None,
}


def set_mqtt_entity_context(client, device_info, ha_prefix):
    _MQTT_ENTITY_CONTEXT["client"] = client
    _MQTT_ENTITY_CONTEXT["device_info"] = device_info
    _MQTT_ENTITY_CONTEXT["ha_prefix"] = ha_prefix


class MqttEntity:
    def __init__(
        self,
        client,
        device_info,
        android_name: str,
        entity_type: str,
        ha_prefix: str,
        enabled: bool = False,
    ):
        self.client = client
        self.device_info = device_info
        self.android_name = android_name
        self.safe_name = self._to_safe_name(android_name)
        self.name = self.android_name
        self.entity_type = entity_type.lower()
        self.unique_id = self.safe_name
        self.config_topic = f"{ha_prefix}/{self.entity_type}/{self.unique_id}/config"
        base_topic = f"{ha_prefix}/{self.unique_id}"
        self.state_topic = f"{base_topic}/state"
        if self.entity_type == "switch":
            self.command_topic = f"{base_topic}/set"
        elif self.entity_type == "button":
            self.command_topic = f"{base_topic}/press"
        else:
            self.command_topic = None
        self.enabled = bool(enabled)

    @staticmethod
    def _to_safe_name(name: str) -> str:
        cleaned = re.sub(r"[^\w]+", "_", name.strip().lower())
        return cleaned.strip("_")

    def publish_discovery(self):
        if self.client is None:
            return

        cfg = {
            "name": self.android_name,
            "unique_id": self.unique_id,
            "device": self.device_info,
        }
        if self.entity_type == "switch":
            cfg.update(
                {
                    "state_topic": self.state_topic,
                    "command_topic": self.command_topic,
                    "payload_on": "ON",
                    "payload_off": "OFF",
                }
            )
        elif self.entity_type == "button":
            cfg.update(
                {
                    "command_topic": self.command_topic,
                    "payload_press": "PRESS",
                }
            )
        elif self.entity_type == "sensor":
            cfg.update(
                {
                    "state_topic": self.state_topic,
                }
            )
        self.client.publish(self.config_topic, json.dumps(cfg), retain=True)
        print(f"✅ Published {self.entity_type} discovery for {self.name}")
        print(cfg)

    def set_state(self, state, force=False):
        if self.entity_type != "switch" or self.client is None:
            return

        if isinstance(state, bool):
            desired = state
            payload = "ON" if state else "OFF"
        else:
            payload = str(state).upper()
            desired = payload == "ON"

        if self.enabled == desired and not force:
            return

        self.enabled = desired
        self.publish_state(payload)
        print(f"💡 {self.name} state -> {payload}")

    def press(self):
        if self.entity_type != "button" or self.client is None:
            return
        self.client.publish(self.command_topic, "PRESS")
        print(f"⚡ {self.name} button pressed")

    def publish_state(self, payload, retain=True):
        if self.client is None:
            return
        self.client.publish(self.state_topic, str(payload), retain=retain)


def _clean_room_text(raw_text: str) -> str:
    """Strip icon glyphs and whitespace from the front of a room label."""
    if not raw_text:
        return ""
    return re.sub(r"^[^A-Za-z0-9]+\s*", "", raw_text).strip()


def _normalize_room_name(name: str) -> str:
    cleaned = _clean_room_text(name).replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _is_room_selected(btn, parent_map) -> bool:
    """Detect selection badge/flags near a room button."""
    def _index_selected(node):
        try:
            return int(node.attrib.get("index", "0")) > 0
        except ValueError:
            return False

    def _flag_selected(node):
        return node.attrib.get("selected") == "true" or node.attrib.get("checked") == "true"

    def _has_number_badge(node):
        for sibling in list(node):
            if sibling is btn:
                continue
            if sibling.attrib.get("class") == "android.widget.TextView":
                txt = (sibling.attrib.get("text", "") or "").strip()
                if txt.isdigit():
                    return True
        return False

    parent = parent_map.get(btn)
    if parent is not None:
        if _flag_selected(parent) or _has_number_badge(parent):
            return True
    return _index_selected(btn) or _flag_selected(btn)


def _get_room_buttons_with_state(tree):
    """Return tuples of (android_name, enabled, button_node)."""
    parent = tree.find(".//*[@resource-id='3d-map-out-div-9527']")
    if parent is None:
        return []

    parent_map = {child: parent for parent in parent.iter() for child in parent}
    rooms = []
    for btn in parent.findall(".//*[@class='android.widget.Button']"):
        raw_text = btn.get("text", "") or ""
        android_name = _clean_room_text(raw_text)
        if not android_name:
            continue
        enabled = _is_room_selected(btn, parent_map)
        rooms.append((android_name, enabled, btn))
    return rooms


def _log_room_debug(room_name, tree):
    """Print debug info for a given room if present in the tree."""
    normalized = _normalize_room_name(room_name)
    parent = tree.find(".//*[@resource-id='3d-map-out-div-9527']")
    if parent is None:
        print("🛑 No map parent found while debugging room state.")
        return
    parent_map = {child: parent for parent in parent.iter() for child in parent}
    for android_name, _, btn in _get_room_buttons_with_state(tree):
        if _normalize_room_name(android_name) != normalized:
            continue
        ancestor = parent_map.get(btn)
        print(
            f"🧭 Debug room '{android_name}': btn(index={btn.attrib.get('index')}, "
            f"selected={btn.attrib.get('selected')}, checked={btn.attrib.get('checked')}, "
            f"bounds={btn.attrib.get('bounds')}), "
            f"parent(index={ancestor.attrib.get('index') if ancestor is not None else None}, "
            f"selected={ancestor.attrib.get('selected') if ancestor is not None else None}, "
            f"checked={ancestor.attrib.get('checked') if ancestor is not None else None})"
        )
        return
    print(f"🛑 Room '{room_name}' not found in debug scan.")


def RefreshRoomState(entities=None):
    navigate_to("Robot")
    refresh_tree()

    button_states = _get_room_buttons_with_state(get_tree())
    if not button_states:
        print("⚠️ No map found for RefreshRoomState()")
        return [] if entities is None else entities

    ctx_client = _MQTT_ENTITY_CONTEXT["client"]
    ctx_device = _MQTT_ENTITY_CONTEXT["device_info"]
    ctx_prefix = _MQTT_ENTITY_CONTEXT["ha_prefix"]
    if None in (ctx_client, ctx_device, ctx_prefix):
        raise RuntimeError(
            "MQTT entity context is not initialized; call set_mqtt_entity_context first."
        )

    if entities is None:
        return [
            MqttEntity(ctx_client, ctx_device, name, "switch", ctx_prefix, enabled)
            for name, enabled, _ in button_states
        ]

    existing = {e.android_name: e for e in entities if e.entity_type == "switch"}
    for name, enabled, _ in button_states:
        entity = existing.get(name)
        if entity is None:
            new_entity = MqttEntity(ctx_client, ctx_device, name, "switch", ctx_prefix, enabled)
            entities.append(new_entity)
            new_entity.publish_discovery()
            new_entity.set_state(enabled, force=True)
            print(f"➕ Added new room entity for {name}")
            continue
        if entity.enabled != enabled:
            entity.set_state(enabled)
        else:
            entity.enabled = enabled
    return entities


# --------------------------
# Enable Rooms
# --------------------------
def enbl_room(room_name):
    navigate_to("Robot")
    refresh_tree()
    tree = get_tree()
    target = None
    normalized = _normalize_room_name(room_name)
    rooms = _get_room_buttons_with_state(tree)
    print(f"🔎 enbl_room searching for '{room_name}' (normalized '{normalized}'). Rooms visible: {[n for n, _, _ in rooms]}")
    for android_name, enabled, btn in rooms:
        if _normalize_room_name(android_name) == normalized:
            target = (android_name, enabled, btn)
            break
    if target is None:
        for android_name, enabled, btn in rooms:
            if normalized in _normalize_room_name(android_name):
                print(f"ℹ️ Using contains-match on '{android_name}'")
                target = (android_name, enabled, btn)
                break
    if target is None and "_" in room_name:
        fallback = room_name.replace("_", " ")
        target = find_by_text(tree, fallback, contains=True)
    if target is None:
        print(f"⚠️ Room '{room_name}' not found on screen.")
        return
    if isinstance(target, tuple):
        android_name, enabled, btn = target
        print(f"➡️ Clicking room '{android_name}' (was enabled={enabled}, bounds={btn.attrib.get('bounds')})")
        click_elem(btn)
    else:
        print(f"➡️ Clicking fallback element for '{room_name}'")
        click_elem(target)
    print(f"🏠 Clicked on room: {room_name}")
    # Give UI a brief moment and log the immediate state seen after click
    sleep(0.3)
    refresh_tree()
    post_state = get_room_enabled_state(room_name)
    print(f"🔁 Post-click state for '{room_name}': {post_state}")

def get_room_enabled_state(room_name):
    """
    Read the current enabled state for a single room button from the UI.
    Returns True/False, or None if the room is not visible.
    """
    tree = get_tree()
    for android_name, enabled, _ in _get_room_buttons_with_state(tree):
        if _normalize_room_name(android_name) == _normalize_room_name(room_name):
            return enabled
    return None

def wait_for_room_state(room_name, desired_state, retries=3, delay=0.5):
    """
    Poll the UI until the room reflects the desired enabled state.
    Returns True on success, False on timeout.
    """
    for attempt in range(1, retries + 1):
        refresh_tree()
        state = get_room_enabled_state(room_name)
        if state is None:
            print(f"⏳ Room '{room_name}' not visible (attempt {attempt}/{retries})")
            _log_room_debug(room_name, get_tree())
        elif state == desired_state:
            print(f"✅ Room '{room_name}' reached state {desired_state} after {attempt} attempt(s)")
            return True
        else:
            print(f"⏳ Room '{room_name}' state {state} != desired {desired_state} (attempt {attempt}/{retries})")
            _log_room_debug(room_name, get_tree())
        sleep(delay)
    print(f"⚠️ Room '{room_name}' did not reach desired state {desired_state} after {retries} attempts")
    return False

# MapScreenshot()

device_info = {
    "identifiers": [DEVICE_NAME.lower().replace(" ", "_")],
    "name": DEVICE_NAME,
    "manufacturer": "PythonMQTT",
    "model": "Robot Vacuum",
}

entities = []
map_status_entity = None
last_map_status = "Unknown"

def mqtt_received(topic, payload):
    decoded_payload = payload.upper()
    print(f"📩 Received '{decoded_payload}' on {topic}")
    handled = False

    for entity in entities:
        if topic != entity.command_topic:
            continue

        handled = True
        if entity.entity_type == "switch":
            desired_state = decoded_payload == "ON"
            if entity.enabled != desired_state:
                print(
                    f"⚙️ Switch {entity.android_name} toggled {decoded_payload} (was {entity.enabled})"
                )
                enbl_room(entity.android_name)
                wait_for_room_state(entity.android_name, desired_state)
            else:
                print(f"ℹ️ {entity.android_name} already {decoded_payload}")
        elif entity.entity_type == "button":
            print(f"⚙️ Button press {entity.name}")
            handler = globals().get(entity.name)
            if callable(handler):
                handler()
            else:
                print(f"⚠️ No handler found for {entity.name}")

    if handled:
        RefreshRoomState(entities)
        MapScreenshot()
        print("🔄 Room state refreshed after command processing.")
    else:
        print(f"⚠️ No entity matched topic {topic}")

# ---------- MQTT CALLBACKS ----------
def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    queue_task(mqtt_received, msg.topic, payload)

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected to MQTT broker with result code", rc)
    for entity in entities:
        if not entity.command_topic:
            continue
        client.subscribe(entity.command_topic)
        print(f"🔔 Subscribed to {entity.command_topic}")

def command_worker():
    while True:
        task = command_queue.get()
        try:
            if callable(task):
                task()
            else:
                topic, payload = task
                print(f"Processing command from {topic}: {payload}")
                mqtt_received(topic, payload)
        except Exception as e:
            print("Error:", e)
        finally:
            command_queue.task_done()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

set_mqtt_entity_context(client, device_info, HA_DISCOVERY_PREFIX)

entities = RefreshRoomState()

map_status_entity = MqttEntity(client, device_info, "Map Status", "sensor", HA_DISCOVERY_PREFIX)
entities.append(map_status_entity)

for name in [n for n in globals() if n.startswith("Click")]:
        entities.append(MqttEntity(client, device_info, name, "button", HA_DISCOVERY_PREFIX))

client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Publish discovery and initial states after the broker connection is established,
# so Home Assistant receives the correct retained states at startup.
for entity in entities:
        entity.publish_discovery()
        if entity.entity_type == "switch":
                entity.set_state(entity.enabled, force=True)

print("🏠 All entities published via MQTT Discovery!")
threading.Thread(target=command_worker, daemon=True).start()
queue_task(map_refresh_task)
client.loop_forever()

# from line_profiler import LineProfiler

# lp = LineProfiler()
# lp.add_function(MapScreenshot)
# lp.runcall(MapScreenshot)
# lp.print_stats()

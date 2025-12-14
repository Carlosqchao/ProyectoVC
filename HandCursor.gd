extends ColorRect

@export var hand_receiver_path: NodePath
@export var hand_label: String = "Left"  # "Left" o "Right"
var hand_receiver: Node

func _ready() -> void:
	mouse_filter = MOUSE_FILTER_IGNORE
	color.a = 0.8  # semitransparente
	if hand_receiver_path != NodePath():
		hand_receiver = get_node(hand_receiver_path)
	else:
		print("HandCursor: hand_receiver_path no asignado")

func _process(_delta: float) -> void:
	if hand_receiver == null:
		return

	var data = hand_receiver.last_data
	if typeof(data) != TYPE_DICTIONARY:
		return
	if not data.has("hands"):
		return

	var hands: Array = data["hands"]
	var found := false

	for hand in hands:
		if hand.has("label") and str(hand["label"]) == hand_label:
			if hand.has("x") and hand.has("y") \
			and hand.has("len_x") and hand.has("len_y") \
			and data.has("w") and data.has("h"):

				var x_px: float = hand["x"]
				var y_px: float = hand["y"]
				var len_x: float = max(float(hand["len_x"]), 20.0)
				var len_y: float = max(float(hand["len_y"]), 20.0)
				var w: float = data["w"]
				var h: float = data["h"]

				var nx = x_px / w
				var ny = y_px / h

				var viewport_size = get_viewport().get_visible_rect().size
				var gx = nx * viewport_size.x
				var gy = ny * viewport_size.y

				global_position = Vector2(gx - len_x * 0.5, gy - len_y * 0.5)
				size = Vector2(len_x, len_y)

				found = true
			break

	visible = found

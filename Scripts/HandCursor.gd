extends ColorRect

@export var hand_receiver_path: NodePath
@export var hand_label: String = "Left"
@export var min_move_distance: float = 5.0  # píxeles

var hand_receiver: Node
var last_global_pos: Vector2

@onready var balde_scene: Node2D = $Balde
@onready var l_scene: Node2D = $L

func _ready() -> void:
	rotation_degrees = 0.0
	scale = Vector2.ONE
	color.a = 0.0

	if hand_receiver_path != NodePath():
		hand_receiver = get_node(hand_receiver_path)
	else:
		print("HandShapeSprite: hand_receiver_path no asignado")

	balde_scene.visible = false
	l_scene.visible = false
	balde_scene.rotation_degrees = 0.0
	l_scene.rotation_degrees = 0.0

	last_global_pos = global_position


func _process(delta: float) -> void:
	if hand_receiver == null:
		return

	var data = hand_receiver.last_data
	if typeof(data) != TYPE_DICTIONARY:
		return
	if not data.has("hands"):
		return

	var hands: Array = data["hands"]
	var viewport_size = get_viewport().get_visible_rect().size
	var found := false

	balde_scene.visible = false
	l_scene.visible = false

	for hand in hands:
		if str(hand.get("label", "")) != hand_label:
			continue

		if not (hand.has("x") and hand.has("y") and hand.has("len_x") and hand.has("len_y")
			and data.has("w") and data.has("h")):
			continue

		var x_px: float = hand["x"]
		var y_px: float = hand["y"]
		var len_x: float = max(float(hand["len_x"]), 20.0)
		var len_y: float = max(float(hand["len_y"]), 20.0)
		var w: float = data["w"]
		var h: float = data["h"]

		var nx = x_px / w
		var ny = y_px / h
		var target_pos = Vector2(nx * viewport_size.x, ny * viewport_size.y)

		# Movimiento con umbral
		var move_vec = target_pos - last_global_pos
		if move_vec.length() >= min_move_distance:
			global_position = target_pos
			last_global_pos = target_pos

		var angle_deg: float = 0.0
		if hand.has("angle"):
			angle_deg = float(hand["angle"])

		var inverted: bool = false
		if hand.has("inverted"):
			inverted = bool(hand["inverted"])

		var shape_name: String = str(hand["shape"])

		match shape_name:
			"rock":
				_update_shape_scene(balde_scene, len_x, len_y, angle_deg, delta, true)
			"L":
				_update_shape_scene_L(l_scene, len_x, len_y, angle_deg, inverted, delta, true)
			_:
				pass

		found = true
		break

	visible = found


func _update_shape_scene(root: Node2D, len_x: float, len_y: float, angle_deg: float, delta: float, hand_detected: bool) -> void:
	var body: CharacterBody2D = root.get_node("CharacterBody2D")
	var sprite: Sprite2D = body.get_node("Sprite2D")

	if sprite.texture:
		var tex_size = sprite.texture.get_size()
		if tex_size.x != 0.0 and tex_size.y != 0.0:
			var scale_x = len_x / tex_size.x
			var scale_y = len_y / tex_size.y
			root.scale = Vector2(scale_x, scale_y)

	root.rotation_degrees = angle_deg - 90.0
	root.visible = true

	if hand_detected and delta > 0.0:
		var desired_pos = root.global_position
		var current_pos = body.global_position
		var displacement = desired_pos - current_pos
		body.external_velocity = (displacement / delta)/5
	# Si no hay mano, no se toca external_velocity


func _update_shape_scene_L(root: Node2D, len_x: float, len_y: float, angle_deg: float, inverted: bool, delta: float, hand_detected: bool) -> void:
	var body: CharacterBody2D = root.get_node("CharacterBody2D")
	var sprite: Sprite2D = body.get_node("Sprite2D")

	if sprite.texture:
		var tex_size = sprite.texture.get_size()
		if tex_size.x != 0.0 and tex_size.y != 0.0:
			var scale_x = len_x / tex_size.x
			var scale_y = len_y / tex_size.y

			if inverted:
				root.scale = Vector2(scale_x, -scale_y)
			else:
				root.scale = Vector2(scale_x, scale_y)

	root.rotation_degrees = angle_deg
	root.visible = true

	if hand_detected and delta > 0.0:
		var desired_pos = root.global_position
		var current_pos = body.global_position
		var displacement = desired_pos - current_pos
		body.external_velocity = displacement / delta
	# Igual: si no hay mano, no se modifica external_velocity

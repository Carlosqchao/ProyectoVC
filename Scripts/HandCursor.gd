extends ColorRect

@export var hand_receiver_path: NodePath
@export var hand_label: String = "Right"
@export var min_move_distance: float = 5.0  # píxeles

var hand_receiver: Node
var last_global_pos: Vector2

@onready var balde_scene: Node2D = $Balde
@onready var palo_scene: Node2D = $Palo
@onready var fuelle_scene: Node2D = $Fuelle
@onready var iman_scene: Node2D = $Iman

func _ready() -> void:
	rotation_degrees = 0.0
	scale = Vector2.ONE
	color.a = 0.0

	if hand_receiver_path != NodePath():
		hand_receiver = get_node(hand_receiver_path)
	else:
		print("HandShapeSprite: hand_receiver_path no asignado")

	balde_scene.visible = false
	palo_scene.visible = false
	fuelle_scene.visible = false
	iman_scene.visible = false
	balde_scene.rotation_degrees = 0.0
	palo_scene.rotation_degrees = 0.0
	fuelle_scene.rotation_degrees = 180.0
	iman_scene.rotation_degrees = 0.0

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
	palo_scene.visible = false
	fuelle_scene.visible = false
	iman_scene.visible = false
	balde_scene.external_velocity = Vector2.ZERO
	palo_scene.external_velocity = Vector2.ZERO
	fuelle_scene.external_velocity = Vector2.ZERO
	iman_scene.external_velocity = Vector2.ZERO

	for hand in hands:
		if str(hand.get("label", "")) != hand_label:
			hand_label = str(hand.get("label", ""))
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

		var shape_name: String = str(hand["shape"])

		match shape_name:
			"rock":
				Globals.fuerzas=false
				_update_shape_scene(balde_scene, len_x, len_y, angle_deg, delta, true)
			"index":
				Globals.fuerzas=false
				_update_shape_scene(palo_scene, len_x, len_y, angle_deg, delta, true)
			"peace":
				Globals.fuerzas=true
				_update_shape_scene(fuelle_scene, len_x, len_y, angle_deg, delta, true)
			"C":
				Globals.fuerzas=true
				_update_shape_scene(iman_scene, len_x, len_y, angle_deg, delta, true)
			_:
				pass

		found = true
		break

	visible = found
	
func _update_shape_scene(body: CharacterBody2D, len_x: float, len_y: float, angle_deg: float, delta: float, hand_detected: bool) -> void:
	body.rotation_degrees = angle_deg - 90.0
	body.visible = hand_detected

	if hand_detected and delta > 0.0:
		var desired_pos: Vector2 = global_position
		var current_pos: Vector2 = body.global_position
		var displacement: Vector2 = desired_pos - current_pos
		body.external_velocity = (displacement / delta) / 5.0
	else:
		# Si esta forma no está activa, que no arrastre nada
		body.external_velocity = Vector2.ZERO

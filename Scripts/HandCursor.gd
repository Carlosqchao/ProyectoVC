extends ColorRect

@export var hand_receiver_path: NodePath
@export var hand_label: String = "Left"

var hand_receiver: Node
@onready var sprite_balde: Sprite2D = $Balde
@onready var sprite_L: Sprite2D = $L

func _ready() -> void:
	rotation_degrees = 0.0
	scale = Vector2.ONE
	color.a = 0.0

	if hand_receiver_path != NodePath():
		hand_receiver = get_node(hand_receiver_path)
	else:
		print("HandShapeSprite: hand_receiver_path no asignado")

	sprite_balde.visible = false
	sprite_L.visible = false
	sprite_balde.rotation_degrees = 0.0
	sprite_L.rotation_degrees = 0.0


func _process(_delta: float) -> void:
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

	sprite_balde.visible = false
	sprite_L.visible = false

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
		global_position = Vector2(nx * viewport_size.x, ny * viewport_size.y)

		var angle_deg: float = 0.0
		if hand.has("angle"):
			angle_deg = float(hand["angle"])

		var inverted: bool = false
		if hand.has("inverted"):
			inverted = bool(hand["inverted"])

		var shape_name: String = str(hand["shape"])

		match shape_name:
			"rock":
				_update_sprite(sprite_balde, len_x, len_y, angle_deg)
			"L":
				_update_sprite_L(sprite_L, len_x, len_y, angle_deg, inverted)
			_:
				pass

		found = true
		break

	visible = found


func _update_sprite(s: Sprite2D, len_x: float, len_y: float, angle_deg: float) -> void:
	if s.texture:
		var tex_size = s.texture.get_size()
		if tex_size.x != 0.0 and tex_size.y != 0.0:
			var scale_x = len_x / tex_size.x
			var scale_y = len_y / tex_size.y
			s.scale = Vector2(scale_x, scale_y)

	s.rotation_degrees = angle_deg - 90.0
	s.visible = true


func _update_sprite_L(s: Sprite2D, len_x: float, len_y: float, angle_deg: float, inverted: bool) -> void:
	if s.texture:
		var tex_size = s.texture.get_size()
		if tex_size.x != 0.0 and tex_size.y != 0.0:
			var scale_x = len_x / tex_size.x
			var scale_y = len_y / tex_size.y
			
			# Si está invertida, voltear verticalmente
			if inverted:
				s.scale = Vector2(scale_x, -scale_y)
			else:
				s.scale = Vector2(scale_x, scale_y)

	s.rotation_degrees = angle_deg
	s.visible = true

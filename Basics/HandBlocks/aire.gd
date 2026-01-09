extends Area2D

@export var strength: float = 1000.0  # Intensidad del aire

@onready var push_dir: Node2D = $"../PushDir"

func _ready() -> void:
	connect("body_entered", _on_body_entered)
	connect("body_exited", _on_body_exited)

func _get_push_vector() -> Vector2:
	# Vector desde el centro del fuelle (Area2D) hasta el nodo de referencia
	return (push_dir.global_position - global_position).normalized()

var bodies_in_wind: Array[RigidBody2D] = []

func _on_body_entered(body: Node) -> void:
	if body is RigidBody2D:
		bodies_in_wind.append(body)

func _on_body_exited(body: Node) -> void:
	if body is RigidBody2D:
		bodies_in_wind.erase(body)
		body.wake_up()

func _physics_process(delta: float) -> void:
	if bodies_in_wind.is_empty():
		return
	var dir := _get_push_vector()
	for body in bodies_in_wind:
		if is_instance_valid(body):
			body.apply_central_force(dir * strength)

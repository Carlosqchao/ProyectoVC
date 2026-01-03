extends CharacterBody2D

var external_velocity: Vector2 = Vector2.ZERO

func _physics_process(_delta: float) -> void:
	velocity = external_velocity
	move_and_slide()

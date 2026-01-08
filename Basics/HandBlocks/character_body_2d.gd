extends CharacterBody2D

var external_velocity: Vector2 = Vector2.ZERO

func _physics_process(_delta: float) -> void:
	for i in get_slide_collision_count():
		var collission:KinematicCollision2D = get_slide_collision(i)
		var collider:Object = collission.get_collider()
		if collider.has_method("push_with_form"):
			collider.push_with_form(-collission.get_normal(),1)

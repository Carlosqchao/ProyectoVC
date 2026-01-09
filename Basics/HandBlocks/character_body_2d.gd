extends CharacterBody2D

var external_velocity: Vector2 = Vector2.ZERO



func _physics_process(_delta: float) -> void:
		# Aplica la velocidad externa (la que le mandas desde el otro script)
	velocity = external_velocity
	# Mover el cuerpo para que se generen colisiones válidas
	move_and_slide()
	for i in get_slide_collision_count():
		var collission:KinematicCollision2D = get_slide_collision(i)
		var collider:Object = collission.get_collider()
		if collider.has_method("push_with_form"):
			var n := -collission.get_normal()
			var force := 400.0  # en vez de 1000
			collider.push_with_form(n, force)

extends RigidBody2D

func wake_up()->void:
	sleeping = false
# Called every frame. 'delta' is the elapsed time since the previous frame.
func push_with_form(dir:Vector2, power:float) -> void:	
	apply_central_force(dir*power)

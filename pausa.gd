extends CanvasLayer


func _physics_process(delta):
	if Input.is_action_just_pressed("pausa"):
		get_tree().paused = not get_tree().paused
		$ColorRect.visible = not $ColorRect.visible
		$VBoxContainer.visible = not $VBoxContainer.visible



func _on_exit_pressed() -> void:
	get_tree().quit()


func _on_level_selector_pressed() -> void:
	get_tree().change_scene_to_file("res://Levels/Level Selector.tscn")
	pass # Replace with function body.

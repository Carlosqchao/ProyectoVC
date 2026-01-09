extends Area2D

@export var Level = "Level1"
# Called when the node enters the scene tree for the first time.
func _on_area_2d_body_entered(body:Node2D)-> void:
	get_tree().change_scene_to_file("res://Levels/"+Level+".tscn")


func _on_body_entered(body: Node2D) -> void:
	pass # Replace with function body.

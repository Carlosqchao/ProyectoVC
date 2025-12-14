extends Node

var udp: PacketPeerUDP = PacketPeerUDP.new()
var last_data: Dictionary = {}

const PORT := 4242

func _ready() -> void:
	var err := udp.bind(PORT)
	if err != OK:
		print("Error al hacer bind UDP:", err)
	else:
		print("Escuchando UDP en puerto", PORT)

func _process(_delta: float) -> void:
	while udp.get_available_packet_count() > 0:
		var bytes: PackedByteArray = udp.get_packet()
		var text := bytes.get_string_from_utf8()
		#print("Paquete recibido:", text)  # puedes dejarlo comentado
		var result = JSON.parse_string(text)
		if typeof(result) == TYPE_DICTIONARY:
			last_data = result

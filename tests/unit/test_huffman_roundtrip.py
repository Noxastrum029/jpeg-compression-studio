from stages.stage3_huffman_coding import HuffmanDecoder, HuffmanEncoder


def test_huffman_encode_decode_roundtrip_with_integer_list():
    values = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, -1, -1]

    encoder = HuffmanEncoder()
    bitstream, code_table = encoder.encode(values)

    decoder = HuffmanDecoder(code_table)
    decoded_values = decoder.decode(bitstream)

    assert decoded_values == values
    assert len(code_table) > 0
    assert len(bitstream) > 0

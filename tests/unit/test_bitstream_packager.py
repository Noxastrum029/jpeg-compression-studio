from pathlib import Path

from stages.stage4_bitstream import BitstreamPackager


def test_jxc_roundtrip(tmp_path, sample_bitstream, sample_code_table, sample_metadata):
    packager = BitstreamPackager()
    output_file = tmp_path / "roundtrip_test.jxc"

    result = packager.create_jxc_file(
        output_path=str(output_file),
        bitstream=sample_bitstream,
        code_table=sample_code_table,
        metadata=sample_metadata,
    )

    assert Path(result["output_path"]).exists()
    assert result["bitstream_length"] == len(sample_bitstream)
    assert result["code_table_entries"] == len(sample_code_table)

    decoded_bitstream, decoded_table, decoded_metadata = packager.read_jxc_file(str(output_file))

    assert decoded_bitstream == sample_bitstream
    assert decoded_table == sample_code_table
    assert decoded_metadata.image_width == sample_metadata.image_width
    assert decoded_metadata.image_height == sample_metadata.image_height
    assert decoded_metadata.wavelet == sample_metadata.wavelet

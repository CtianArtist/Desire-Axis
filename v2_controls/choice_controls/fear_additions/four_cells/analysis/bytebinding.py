"""Preserve selected original lexical bytes without exporting unrelated rows."""
import csv
import io
import json
from .core import digest, sha256


def bind_csv(source, output, selected):
    text = source.read_bytes().decode('utf-8')
    lines = text.splitlines(keepends=True)
    reader = csv.DictReader(io.StringIO(text, newline=''))
    reader.fieldnames  # consume original header
    previous = reader.line_num
    pieces = [''.join(lines[:previous])]
    wanted = {digest(r) for r in selected}
    bindings = []
    for row in reader:
        end = reader.line_num
        if digest(row) in wanted:
            piece = ''.join(lines[previous:end])
            pieces.append(piece)
            bindings.append({'source_lines': [previous + 1, end], 'row_sha256': digest(row)})
        previous = end
    if len(bindings) != len(selected):
        raise ValueError('CSV byte binding selection mismatch')
    output.write_bytes(''.join(pieces).encode('utf-8'))
    return {'source_sha256': sha256(source), 'selected_bytes_sha256': sha256(output), 'rows': bindings}


def bind_json(source, output, selected):
    text = source.read_bytes().decode('utf-8')
    decoder = json.JSONDecoder()
    def skip(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    i = skip(0)
    if text[i] == '{':
        i = skip(i + 1)
        while True:
            key, end = decoder.raw_decode(text, i)
            i = skip(end)
            if text[i] != ':':
                raise ValueError('invalid JSON object')
            i = skip(i + 1)
            if key == 'rates':
                break
            _, i = decoder.raw_decode(text, i)
            i = skip(i)
            if text[i] != ',':
                raise ValueError('missing rates array')
            i = skip(i + 1)
    if text[i] != '[':
        raise ValueError('expected rate array')
    i = skip(i + 1)
    wanted = {digest(r) for r in selected}
    pieces, bindings = [], []
    while text[i] != ']':
        row, end = decoder.raw_decode(text, i)
        if digest(row) in wanted:
            pieces.append(text[i:end])
            bindings.append({'source_byte_start': len(text[:i].encode('utf-8')),
                             'source_byte_end': len(text[:end].encode('utf-8')), 'row_sha256': digest(row)})
        i = skip(end)
        if text[i] == ',':
            i = skip(i + 1)
        elif text[i] != ']':
            raise ValueError('invalid rate array')
    if len(bindings) != len(selected):
        raise ValueError('JSON byte binding selection mismatch')
    output.write_bytes(('[\n' + ',\n'.join(pieces) + '\n]\n').encode('utf-8'))
    return {'source_sha256': sha256(source), 'selected_bytes_sha256': sha256(output), 'rows': bindings}

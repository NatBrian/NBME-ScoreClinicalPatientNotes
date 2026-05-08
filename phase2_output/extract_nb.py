import json, base64, os

with open('phase2_output/phase2_lora_training_report.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

out_dir = 'phase2_output/nb_extracted'
os.makedirs(out_dir, exist_ok=True)

cells = nb['cells']
text_lines = []

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        text_lines.append(f'=== MARKDOWN CELL {i} ===')
        text_lines.append(src)
        text_lines.append('')
    elif cell['cell_type'] == 'code':
        if 'outputs' in cell:
            for j, o in enumerate(cell['outputs']):
                ot = o.get('output_type','')
                if ot == 'stream':
                    text = ''.join(o.get('text', []))
                    text_lines.append(f'=== CODE CELL {i} STREAM OUTPUT ===')
                    text_lines.append(text[:5000])
                    text_lines.append('')
                elif ot in ('display_data', 'execute_result'):
                    data = o.get('data', {})
                    if 'image/png' in data:
                        img_data = data['image/png']
                        fname = f'{out_dir}/cell{i:02d}_out{j}.png'
                        with open(fname, 'wb') as f2:
                            f2.write(base64.b64decode(img_data))
                        text_lines.append(f'=== CODE CELL {i} IMAGE saved to {fname} ===')
                    if 'text/html' in data:
                        html = ''.join(data['text/html'])
                        text_lines.append(f'=== CODE CELL {i} HTML TABLE ===')
                        text_lines.append(html[:5000])
                        text_lines.append('')

with open(f'{out_dir}/extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(text_lines))

print('Done. Files saved to', out_dir)
print('Images:')
for fn in os.listdir(out_dir):
    if fn.endswith('.png'):
        print(' ', fn)

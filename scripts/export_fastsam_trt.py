from ultralytics import FastSAM
from daaam import ROOT_DIR
import os
import click

@click.command()
@click.option('--imgsz', nargs=2, type=int, default=[480, 640], help='Image size (h w)')
@click.option('--half', is_flag=True, help='Export FP16 engine')
@click.option('--simplify', is_flag=True, help='Simplify ONNX model')
@click.option('--batch', type=int, default=1, help='Batch size for export')
@click.option('--model_name', type=str, default='FastSAM-x', help='Model name for output files')
@click.option('--device', default='cuda:0', help='CUDA device for export')
def main(imgsz, half, simplify, batch, model_name, device):
    os.makedirs(f'{ROOT_DIR}/checkpoints/fastsam', exist_ok=True)
    m = FastSAM(f'{ROOT_DIR}/checkpoints/fastsam/{model_name}.pt')
    m.export(format='engine', imgsz=imgsz, half=half, simplify=simplify, batch=batch, opset=18, device=device)
    os.rename(f'{ROOT_DIR}/checkpoints/fastsam/{model_name}.engine', f'{ROOT_DIR}/checkpoints/fastsam/{model_name}-{imgsz[1]}x{imgsz[0]}.engine')

if __name__ == "__main__":
    main()
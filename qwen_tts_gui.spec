# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Qwen3 TTS GUI Application
Comprehensive version that automatically collects all required modules
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

# Automatically collect all submodules from key packages
hiddenimports = []

# Collect all qwen_tts submodules
try:
    qwen_submodules = collect_submodules('qwen_tts')
    hiddenimports.extend(qwen_submodules)
except:
    pass

# Collect all transformers submodules
try:
    transformers_submodules = collect_submodules('transformers')
    hiddenimports.extend(transformers_submodules)
except:
    pass

# Collect all torch submodules (this is large, but necessary)
try:
    torch_submodules = collect_submodules('torch')
    hiddenimports.extend(torch_submodules)
except:
    pass

# Collect all torchvision submodules
try:
    torchvision_submodules = collect_submodules('torchvision')
    hiddenimports.extend(torchvision_submodules)
except:
    pass

# Manual additions for critical modules
critical_imports = [
    # Standard library
    'threading',
    'os',
    'unittest',  # Required by PyTorch
    'importlib',
    'importlib.util',
    
    # Tkinter
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    
    # Core dependencies
    'numpy',
    'sounddevice',
    'soundfile',
    'scipy',
    
    # PIL/Pillow
    'PIL',
    'PIL.Image',
    'PIL.ImageFile',
    'PIL._tkinter_finder',
    
    # PyTorch core
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.serialization',
    'torch.fx',
    'torch.fx.experimental',
    'torch.fx.experimental._config',
    'torch.utils._config_module',
    'torch.nested',
    'torch.nested._internal',
    'torch.nested._internal.nested_tensor',
    'torch.nested._internal.nested_int',
    
    # Transformers
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'transformers.models.auto.processing_auto',
    'transformers.utils',
    'transformers.utils.import_utils',
    'transformers.video_processing_utils',
    'transformers.image_processing_utils_fast',
    
    # SentencePiece
    'sentencepiece',
    'sentencepiece.sentencepiece_pb2',
    
    # Protocol Buffers
    'google.protobuf',
    'google.protobuf.message',
    'google.protobuf.descriptor',
    
    # ONNX Runtime
    'onnxruntime',
    
    # Qwen TTS specific
    'qwen_tts',
    'qwen_tts.inference',
    'qwen_tts.inference.qwen3_tts_model',
    
    # Additional transformers dependencies
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'accelerate',
    
    # Additional torch utilities
    'torch.utils',
    'torch.utils.data',
    'torch.utils.model_zoo',
    'torch._C',
    'torch._C._nn',
    'torch.backends',
    'torch.backends.cuda',
    'torch.backends.cudnn',
]

hiddenimports.extend(critical_imports)

# Remove duplicates while preserving order
seen = set()
hiddenimports = [x for x in hiddenimports if not (x in seen or seen.add(x))]

# Collect data files
datas = []

# Collect qwen_tts data files
try:
    qwen_datas = collect_data_files('qwen_tts')
    datas.extend(qwen_datas)
except:
    pass

# Collect transformers data files
try:
    transformers_datas = collect_data_files('transformers')
    datas.extend(transformers_datas)
except:
    pass

# Collect torch data files
try:
    torch_datas = collect_data_files('torch')
    datas.extend(torch_datas)
except:
    pass

# Collect sentencepiece data files
try:
    sentencepiece_datas = collect_data_files('sentencepiece')
    datas.extend(sentencepiece_datas)
except:
    pass

# Collect binaries (DLLs, shared libraries)
binaries = []

# Try to collect all binaries from torch (CUDA libraries, etc.)
try:
    torch_all = collect_all('torch')
    if 'binaries' in torch_all:
        binaries.extend(torch_all['binaries'])
    if 'datas' in torch_all:
        datas.extend(torch_all['datas'])
except:
    pass

a = Analysis(
    ['src/qwen_tts_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only exclude truly unnecessary packages
        'matplotlib',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        'tkinter.test',
        # Don't exclude unittest, PIL, test modules as they're needed
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='QwenTTS_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging, False for windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if you have one
)

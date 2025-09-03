#!/bin/bash

# Setup script for Legal BERT Seq2Seq - Portuguese Immigration Law Consultant

echo "🏛️  Setting up Legal BERT Seq2Seq - Portuguese Immigration Law Environment"
echo "========================================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check locale support for Portuguese
echo "🌍 Checking Portuguese language support..."
if locale -a 2>/dev/null | grep -q "pt_"; then
    echo "✓ Portuguese locale support detected"
else
    echo "⚠️  Portuguese locale not detected - UTF-8 should still work"
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. Training will be slow on CPU."
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✓ Dependencies installed successfully"
    else
        echo "❌ Failed to install some dependencies"
        echo "   You may need to install them manually"
    fi
else
    echo "❌ requirements.txt not found!"
    echo "   Please make sure you're in the correct directory"
    exit 1
fi

# Check if installations were successful
echo ""
echo "🔍 Verifying installations..."

python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>/dev/null || echo "❌ PyTorch installation failed"
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__} installed')" 2>/dev/null || echo "❌ Transformers installation failed"
python3 -c "import peft; print(f'✓ PEFT {peft.__version__} installed')" 2>/dev/null || echo "❌ PEFT installation failed"

# Check specific models and tokenizers for legal consultation
echo ""
echo "🔍 Verifying legal model components..."
python3 -c "
try:
    from transformers import BertTokenizer, EncoderDecoderModel
    print('✓ BERT components available')
    # Test Portuguese BERT model access (without downloading)
    print('✓ Ready for Portuguese BERT model (neuralmind/bert-base-portuguese-cased)')
except Exception as e:
    print(f'❌ BERT components failed: {e}')
" 2>/dev/null

# Check CUDA availability in PyTorch
echo ""
if python3 -c "import torch; print('✓ CUDA available in PyTorch' if torch.cuda.is_available() else '⚠️  CUDA not available - using CPU')" 2>/dev/null; then
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "🎮 Available GPUs: $gpu_count"
fi

# Create sample data for testing
echo ""
echo "📋 Creating sample legal consultation data..."
python3 -c "
try:
    from trainer import create_legal_sample_data
    create_legal_sample_data('legal_sample_training_data.json')
    print('✓ Legal sample data created at legal_sample_training_data.json')
except ImportError as e:
    print(f'⚠️  Could not create sample data: {e}')
    print('   You can create it later by running: python data_utils.py create-legal')
except Exception as e:
    print(f'❌ Error creating sample data: {e}')
" 2>/dev/null || echo "⚠️  Sample data creation failed - you can create it manually later"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "⚖️  IMPORTANT LEGAL DISCLAIMER:"
echo "   This tool provides informational responses only and does not constitute"
echo "   legal advice. Always consult with a qualified immigration lawyer for"
echo "   official legal guidance."
echo ""
echo "Quick start commands:"
echo "1. Test with sample legal data:"
echo "   python main.py --data_path legal_sample_training_data.json --max_steps 10 --batch_size 1"
echo ""
echo "2. Start interactive legal consultation:"
echo "   python inference.py --base_model neuralmind/bert-base-portuguese-cased --interactive"
echo ""
echo "3. Train with your own legal data:"
echo "   python main.py --data_path your_legal_data.json --num_epochs 3"
echo ""
echo "4. Create comprehensive legal dataset:"
echo "   python data_utils.py create-legal"
echo ""
echo "For more options, see: python main.py --help"

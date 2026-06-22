@echo off
chcp 65001 >nul
cls

echo =========================================
echo Enterprise AI Assistant - Quick Start
echo =========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker found
echo.

REM Create necessary directories
echo 📁 Creating directories...
if not exist "models\sentiment" mkdir models\sentiment
if not exist "models\lora-qwen3b-amazon-valid" mkdir models\lora-qwen3b-amazon-valid
if not exist "chroma_store" mkdir chroma_store
if not exist "training\examples" mkdir training\examples
echo ✅ Directories created
echo.

REM Build and start the application
echo 🔨 Building Docker image...
docker-compose build

echo.
echo 🚀 Starting the application...
docker-compose up -d

echo.
echo =========================================
echo ✅ Application is starting!
echo =========================================
echo.
echo 📊 Access the application:
echo    - API: http://localhost:8000
echo    - Metrics: http://localhost:8000/metrics
echo.
echo 📝 View logs:
echo    docker-compose logs -f
echo.
echo 🛑 Stop the application:
echo    docker-compose down
echo.
echo ⏳ First run will download the 3B model (~6GB). This may take a few minutes...
echo    Check progress with: docker-compose logs -f
echo.
pause
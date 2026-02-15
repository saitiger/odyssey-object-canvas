# StoryBoardai - Getting Started

An AI-powered video generation app that connects to the Odyssey API for real-time video streaming with intelligent scene breakdown.

## Features

- 🎬 **Story Input** - Enter a high-level story idea with optional style presets (Action, Cinematic, Noir, etc.)
- 🤖 **AI Scene Breakdown** - Claude Haiku automatically segments your idea into timed scene prompts
- 🎥 **Live Video Generation** - Stream video in real-time via Odyssey API with sequential prompt injection
- 🖼️ **Image-to-Video** - Optionally attach a reference image for image-based video generation
- ✏️ **Editable Storyboard** - Review and modify generated scenes before filming

## Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- [Odyssey API Key](https://odyssey.ml) - For video generation
- [Anthropic API Key](https://console.anthropic.com/settings/keys) - For AI scene breakdown (optional, falls back to local generation)

## Installation

1. **Clone or download the repository**

   ```bash
   git clone <repository-url>
   cd odyssey-simulation
   ```

2. **Install dependencies**

   ```bash
   cd js
   npm install
   ```

## Running the App

1. **Start the development server**

   ```bash
   cd js
   npm run dev
   ```

2. **Open in browser**

   Navigate to `http://localhost:5173` (or the URL shown in terminal)

## Tech Stack

- **Frontend**: Vanilla JavaScript (no framework)
- **Styling**: CSS with custom styles
- **Build Tool**: Vite
- **APIs**: Odyssey SDK, Anthropic Claude API

## Troubleshooting

### "Connection failed" error
- Check that your Odyssey API key is valid
- Ensure you don't have another session open (max 1 concurrent connection)
- Wait 40 seconds if a previous session wasn't properly disconnected

### AI scene generation not working
- Check browser console for errors
- Verify your Anthropic API key is set correctly
- The app will fall back to local scene generation if AI fails

### Image upload errors
- Ensure image is under 25MB
- Use supported formats: JPEG, PNG, WebP, GIF, BMP
- Check that the file is a real image (not renamed from another format)

## License

MIT

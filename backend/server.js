const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Bot status endpoint
app.get('/api/bot-status/:mode', async (req, res) => {
  const { mode } = req.params;
  try {
    const status = mode === 'demo' ? 
      await BotManager.get_demo_bot_status() : 
      await BotManager.get_live_bot_status();
    res.json({ status: 'success', data: status });
  } catch (error) {
    res.status(400).json({ status: 'error', message: error.message });
  }
});

// Start bot endpoint
app.post('/api/start-bot/:mode', async (req, res) => {
  const { mode } = req.params;
  try {
    await BotManager.start_bots();
    res.json({ status: 'success' });
  } catch (error) {
    res.status(400).json({ status: 'error', message: error.message });
  }
});

// Stop bot endpoint
app.post('/api/stop-bot/:mode', async (req, res) => {
  const { mode } = req.params;
  try {
    await BotManager.stop_bots();
    res.json({ status: 'success' });
  } catch (error) {
    res.status(400).json({ status: 'error', message: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

import { useConversation } from '@elevenlabs/react';
import { useState } from 'react';
import { Button } from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';

export default function DashboardPage() {
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const { status, startSession, stopSession } = useConversation();
  
  const handleVoiceToggle = async () => {
    if (status === 'connected') {
      stopSession();
      return;
    }

    try {
      const response = await fetch('/api/voice/token');
      if (!response.ok) throw new Error('Failed to get voice token');
      
      const { signed_url } = await response.json();
      await startSession({ signedUrl: signed_url });
    } catch (err) {
      setVoiceError('Voice session failed to start. Please try again.');
      console.error('Voice session error:', err);
    }
  };

  return (
    <div>
      {/* Your existing dashboard UI */}
      
      <Button
        variant="contained"
        color={status === 'connected' ? 'error' : 'primary'}
        startIcon={status === 'connected' ? <MicOffIcon /> : <MicIcon />}
        onClick={handleVoiceToggle}
        disabled={status === 'connecting'}
      >
        {status === 'connected' ? 'End Voice Analyst' : 'Start Voice Analyst'}
      </Button>
      
      {voiceError && (
        <div className="error-message">
          {voiceError}
        </div>
      )}
    </div>
  );
}

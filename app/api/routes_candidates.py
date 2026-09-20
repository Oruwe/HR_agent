from fastapi import APIRouter, HTTPException, Depends
from elevenlabs import ElevenLabs
from app.config import get_settings

router = APIRouter()

@router.get("/voice/token")
async def get_elevenlabs_token():
    """Get a signed URL for ElevenLabs Conversational AI voice session"""
    settings = get_settings()
    
    if not settings.elevenlabs_api_key or not settings.elevenlabs_agent_id:
        raise HTTPException(
            status_code=501,
            detail="ElevenLabs voice integration not configured"
        )
    
    try:
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        response = client.convai.get_signed_url(
            agent_id=settings.elevenlabs_agent_id
        )
        return {"signed_url": response.url}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ElevenLabs token: {str(e)}"
        )

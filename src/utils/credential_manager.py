"""
Secure credential storage system for proxy authentication and API keys.
Provides encryption, secure storage, and credential rotation capabilities.
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class CredentialManager:
    """Manages secure storage and retrieval of sensitive credentials."""
    
    def __init__(self, master_key: Optional[str] = None, storage_path: str = "credentials.enc"):
        """
        Initialize credential manager.
        
        Args:
            master_key: Master key for encryption (will be derived if not provided)
            storage_path: Path to encrypted credential storage file
        """
        self.storage_path = storage_path
        self._cipher = None
        self._credentials = {}
        
        # Initialize encryption key
        if master_key:
            self._init_cipher_from_key(master_key)
        else:
            self._init_cipher_from_env()
        
        # Load existing credentials
        self._load_credentials()
    
    def _init_cipher_from_key(self, master_key: str):
        """Initialize cipher from provided master key."""
        # Derive key from master key
        salt = b'youtube_summarizer_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._cipher = Fernet(key)
    
    def _init_cipher_from_env(self):
        """Initialize cipher from environment variables."""
        # Try to get encryption key from environment
        env_key = os.environ.get('ENCRYPTION_KEY')
        if env_key:
            self._init_cipher_from_key(env_key)
        else:
            # Generate new key if none exists
            key = Fernet.generate_key()
            self._cipher = Fernet(key)
            logger.warning("No encryption key provided, generated new key. Set ENCRYPTION_KEY environment variable.")
    
    def _load_credentials(self):
        """Load encrypted credentials from storage."""
        if not os.path.exists(self.storage_path):
            self._credentials = {}
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                encrypted_data = f.read()
            
            if encrypted_data:
                decrypted_data = self._cipher.decrypt(encrypted_data)
                self._credentials = json.loads(decrypted_data.decode())
            else:
                self._credentials = {}
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            self._credentials = {}
    
    def _save_credentials(self):
        """Save encrypted credentials to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Encrypt and save credentials
            data = json.dumps(self._credentials).encode()
            encrypted_data = self._cipher.encrypt(data)
            
            with open(self.storage_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.storage_path, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
    
    def store_credential(self, category: str, key: str, value: str, metadata: Optional[Dict] = None):
        """
        Store a credential securely.
        
        Args:
            category: Category of credential (e.g., 'proxy', 'api_key')
            key: Unique identifier for the credential
            value: The credential value
            metadata: Additional metadata (creation time, expiry, etc.)
        """
        if category not in self._credentials:
            self._credentials[category] = {}
        
        credential_data = {
            'value': value,
            'created_at': secrets.token_hex(8),  # Use as timestamp replacement
            'metadata': metadata or {}
        }
        
        self._credentials[category][key] = credential_data
        self._save_credentials()
        
        logger.info(f"Stored credential: {category}.{key}")
    
    def get_credential(self, category: str, key: str) -> Optional[str]:
        """
        Retrieve a credential value.
        
        Args:
            category: Category of credential
            key: Unique identifier for the credential
            
        Returns:
            The credential value or None if not found
        """
        if category not in self._credentials:
            return None
        
        if key not in self._credentials[category]:
            return None
        
        return self._credentials[category][key]['value']
    
    def get_credential_metadata(self, category: str, key: str) -> Optional[Dict]:
        """
        Retrieve credential metadata.
        
        Args:
            category: Category of credential
            key: Unique identifier for the credential
            
        Returns:
            Metadata dictionary or None if not found
        """
        if category not in self._credentials:
            return None
        
        if key not in self._credentials[category]:
            return None
        
        return self._credentials[category][key]['metadata']
    
    def delete_credential(self, category: str, key: str) -> bool:
        """
        Delete a credential.
        
        Args:
            category: Category of credential
            key: Unique identifier for the credential
            
        Returns:
            True if deleted, False if not found
        """
        if category not in self._credentials:
            return False
        
        if key not in self._credentials[category]:
            return False
        
        del self._credentials[category][key]
        self._save_credentials()
        
        logger.info(f"Deleted credential: {category}.{key}")
        return True
    
    def list_credentials(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all stored credentials.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary mapping categories to lists of credential keys
        """
        if category:
            if category in self._credentials:
                return {category: list(self._credentials[category].keys())}
            else:
                return {}
        
        return {cat: list(keys.keys()) for cat, keys in self._credentials.items()}
    
    def rotate_credential(self, category: str, key: str, new_value: str):
        """
        Rotate a credential (store new value and mark old one for cleanup).
        
        Args:
            category: Category of credential
            key: Unique identifier for the credential
            new_value: New credential value
        """
        # Store old value with rotation suffix
        old_value = self.get_credential(category, key)
        if old_value:
            old_key = f"{key}_old"
            self.store_credential(category, old_key, old_value, {'rotated': True})
        
        # Store new value
        self.store_credential(category, key, new_value, {'rotated_at': secrets.token_hex(8)})
        
        logger.info(f"Rotated credential: {category}.{key}")
    
    def cleanup_rotated_credentials(self, category: Optional[str] = None):
        """
        Clean up old rotated credentials.
        
        Args:
            category: Optional category filter
        """
        categories = [category] if category else list(self._credentials.keys())
        
        for cat in categories:
            if cat not in self._credentials:
                continue
            
            keys_to_delete = []
            for key, data in self._credentials[cat].items():
                if data.get('metadata', {}).get('rotated'):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._credentials[cat][key]
                logger.info(f"Cleaned up rotated credential: {cat}.{key}")
        
        if keys_to_delete:
            self._save_credentials()
    
    def export_credentials(self, category: str, include_values: bool = False) -> Dict[str, Any]:
        """
        Export credentials for backup or migration.
        
        Args:
            category: Category to export
            include_values: Whether to include actual credential values
            
        Returns:
            Dictionary with credential data
        """
        if category not in self._credentials:
            return {}
        
        exported = {}
        for key, data in self._credentials[category].items():
            exported[key] = {
                'metadata': data['metadata'],
                'created_at': data['created_at']
            }
            if include_values:
                exported[key]['value'] = data['value']
        
        return exported
    
    def import_credentials(self, category: str, credentials: Dict[str, Any]):
        """
        Import credentials from backup or migration.
        
        Args:
            category: Category to import to
            credentials: Dictionary with credential data
        """
        if category not in self._credentials:
            self._credentials[category] = {}
        
        for key, data in credentials.items():
            if 'value' in data:
                self._credentials[category][key] = {
                    'value': data['value'],
                    'created_at': data.get('created_at', secrets.token_hex(8)),
                    'metadata': data.get('metadata', {})
                }
        
        self._save_credentials()
        logger.info(f"Imported {len(credentials)} credentials to category: {category}")
    
    def validate_credential_integrity(self) -> bool:
        """
        Validate the integrity of stored credentials.
        
        Returns:
            True if all credentials are valid, False otherwise
        """
        try:
            # Try to decrypt and parse all credentials
            with open(self.storage_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            test_credentials = json.loads(decrypted_data.decode())
            
            # Basic structure validation
            for category, creds in test_credentials.items():
                if not isinstance(creds, dict):
                    return False
                for key, data in creds.items():
                    if not isinstance(data, dict) or 'value' not in data:
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Credential integrity validation failed: {e}")
            return False
    
    def get_proxy_credentials(self) -> Optional[Dict[str, str]]:
        """
        Get proxy credentials in format expected by proxy manager.
        
        Returns:
            Dictionary with username and password or None
        """
        username = self.get_credential('proxy', 'username')
        password = self.get_credential('proxy', 'password')
        
        if username and password:
            return {'username': username, 'password': password}
        
        return None
    
    def store_proxy_credentials(self, username: str, password: str):
        """
        Store proxy credentials.
        
        Args:
            username: Proxy username
            password: Proxy password
        """
        self.store_credential('proxy', 'username', username)
        self.store_credential('proxy', 'password', password)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            API key or None if not found
        """
        return self.get_credential('api_keys', provider)
    
    def store_api_key(self, provider: str, api_key: str):
        """
        Store API key for a specific provider.
        
        Args:
            provider: Provider name
            api_key: API key value
        """
        self.store_credential('api_keys', provider, api_key)


# Global credential manager instance
_credential_manager = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        # Use credentials directory in project root
        storage_path = os.path.join(os.path.dirname(__file__), '..', '..', 'credentials', 'credentials.enc')
        _credential_manager = CredentialManager(storage_path=storage_path)
    return _credential_manager


def secure_credential_fallback(env_var: str, category: str, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get credential with fallback to environment variable.
    
    Args:
        env_var: Environment variable name
        category: Credential category
        key: Credential key
        default: Default value if not found
        
    Returns:
        Credential value or default
    """
    # Try credential manager first
    cm = get_credential_manager()
    value = cm.get_credential(category, key)
    if value:
        return value
    
    # Fall back to environment variable
    value = os.environ.get(env_var)
    if value:
        # Store in credential manager for future use
        cm.store_credential(category, key, value)
        return value
    
    return default
import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { authApi } from '../api/client'
import {
  User,
  Lock,
  Mail,
  Shield,
  FlaskConical,
  UserCircle,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react'

export default function Settings() {
  const { user } = useAuth()
  const [activeTab, setActiveTab] = useState<'profile' | 'password'>('profile')

  // Password change state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [passwordLoading, setPasswordLoading] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <Shield className="h-5 w-5 text-red-500" />
      case 'researcher':
        return <FlaskConical className="h-5 w-5 text-blue-500" />
      default:
        return <UserCircle className="h-5 w-5 text-green-500" />
    }
  }

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-red-100 text-red-700'
      case 'researcher':
        return 'bg-blue-100 text-blue-700'
      default:
        return 'bg-green-100 text-green-700'
    }
  }

  const getTierBadgeColor = (tier: string) => {
    switch (tier?.toLowerCase()) {
      case 'gold':
        return 'bg-yellow-100 text-yellow-700'
      case 'silver':
        return 'bg-gray-100 text-gray-700'
      default:
        return 'bg-orange-100 text-orange-700'
    }
  }

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordMessage(null)

    // Validation
    if (!currentPassword || !newPassword || !confirmPassword) {
      setPasswordMessage({ type: 'error', text: 'All fields are required' })
      return
    }

    if (newPassword.length < 8) {
      setPasswordMessage({ type: 'error', text: 'New password must be at least 8 characters' })
      return
    }

    if (newPassword !== confirmPassword) {
      setPasswordMessage({ type: 'error', text: 'New passwords do not match' })
      return
    }

    setPasswordLoading(true)
    try {
      await authApi.changePassword(currentPassword, newPassword)
      setPasswordMessage({ type: 'success', text: 'Password changed successfully!' })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Failed to change password'
      setPasswordMessage({ type: 'error', text: message })
    } finally {
      setPasswordLoading(false)
    }
  }

  const getPasswordStrength = (password: string): { strength: number; label: string; color: string } => {
    let strength = 0
    if (password.length >= 8) strength++
    if (password.length >= 12) strength++
    if (/[A-Z]/.test(password)) strength++
    if (/[0-9]/.test(password)) strength++
    if (/[^A-Za-z0-9]/.test(password)) strength++

    if (strength <= 1) return { strength: 1, label: 'Weak', color: 'bg-red-500' }
    if (strength <= 2) return { strength: 2, label: 'Fair', color: 'bg-orange-500' }
    if (strength <= 3) return { strength: 3, label: 'Good', color: 'bg-yellow-500' }
    if (strength <= 4) return { strength: 4, label: 'Strong', color: 'bg-green-500' }
    return { strength: 5, label: 'Very Strong', color: 'bg-green-600' }
  }

  const passwordStrength = getPasswordStrength(newPassword)

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      {/* Tabs */}
      <div className="flex gap-1 border-b mb-6">
        <button
          onClick={() => setActiveTab('profile')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'profile'
              ? 'text-primary border-b-2 border-primary'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <User className="h-4 w-4 inline mr-2" />
          Profile
        </button>
        <button
          onClick={() => setActiveTab('password')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'password'
              ? 'text-primary border-b-2 border-primary'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <Lock className="h-4 w-4 inline mr-2" />
          Password
        </button>
      </div>

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <div className="bg-card border rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-6">Profile Information</h2>

          <div className="flex items-start gap-6">
            {/* Avatar */}
            <div className="w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
              <User className="h-12 w-12 text-primary" />
            </div>

            {/* Info */}
            <div className="flex-1 space-y-4">
              {/* Username */}
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Username</label>
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span className="text-lg font-medium">{user?.username}</span>
                </div>
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Email</label>
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span>{user?.email}</span>
                </div>
              </div>

              {/* Role */}
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Role</label>
                <div className="flex items-center gap-2">
                  {user && getRoleIcon(user.role)}
                  <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${
                    user && getRoleBadgeColor(user.role)
                  }`}>
                    {user?.role}
                  </span>
                </div>
              </div>

              {/* Rater Tier (if applicable) */}
              {user?.rater_tier && (
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">Rater Tier</label>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${
                    getTierBadgeColor(user.rater_tier)
                  }`}>
                    {user.rater_tier}
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 pt-6 border-t">
            <p className="text-sm text-muted-foreground">
              To change your username or email, please contact an administrator.
            </p>
          </div>
        </div>
      )}

      {/* Password Tab */}
      {activeTab === 'password' && (
        <div className="bg-card border rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-6">Change Password</h2>

          {passwordMessage && (
            <div className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${
              passwordMessage.type === 'success'
                ? 'bg-green-50 text-green-700 border border-green-200'
                : 'bg-red-50 text-red-700 border border-red-200'
            }`}>
              {passwordMessage.type === 'success' ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <AlertCircle className="h-5 w-5" />
              )}
              {passwordMessage.text}
            </div>
          )}

          <form onSubmit={handlePasswordChange} className="space-y-4 max-w-md">
            {/* Current Password */}
            <div>
              <label className="block text-sm font-medium mb-1">Current Password</label>
              <div className="relative">
                <input
                  type={showCurrentPassword ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg pr-10 focus:outline-none focus:ring-2 focus:ring-primary/50"
                  placeholder="Enter current password"
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div>
              <label className="block text-sm font-medium mb-1">New Password</label>
              <div className="relative">
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg pr-10 focus:outline-none focus:ring-2 focus:ring-primary/50"
                  placeholder="Enter new password"
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {/* Password Strength Indicator */}
              {newPassword && (
                <div className="mt-2">
                  <div className="flex gap-1 mb-1">
                    {[1, 2, 3, 4, 5].map((level) => (
                      <div
                        key={level}
                        className={`h-1 flex-1 rounded ${
                          level <= passwordStrength.strength ? passwordStrength.color : 'bg-gray-200'
                        }`}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Password strength: <span className="font-medium">{passwordStrength.label}</span>
                  </p>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-sm font-medium mb-1">Confirm New Password</label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg pr-10 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                    confirmPassword && confirmPassword !== newPassword ? 'border-red-300' : ''
                  }`}
                  placeholder="Confirm new password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {confirmPassword && confirmPassword !== newPassword && (
                <p className="text-xs text-red-500 mt-1">Passwords do not match</p>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={passwordLoading || !currentPassword || !newPassword || !confirmPassword || newPassword !== confirmPassword}
              className="w-full py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {passwordLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Changing Password...
                </>
              ) : (
                <>
                  <Lock className="h-4 w-4" />
                  Change Password
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t">
            <h3 className="text-sm font-medium mb-2">Password Requirements</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li className={newPassword.length >= 8 ? 'text-green-600' : ''}>
                {newPassword.length >= 8 ? '✓' : '•'} At least 8 characters
              </li>
              <li className={/[A-Z]/.test(newPassword) ? 'text-green-600' : ''}>
                {/[A-Z]/.test(newPassword) ? '✓' : '•'} One uppercase letter (recommended)
              </li>
              <li className={/[0-9]/.test(newPassword) ? 'text-green-600' : ''}>
                {/[0-9]/.test(newPassword) ? '✓' : '•'} One number (recommended)
              </li>
              <li className={/[^A-Za-z0-9]/.test(newPassword) ? 'text-green-600' : ''}>
                {/[^A-Za-z0-9]/.test(newPassword) ? '✓' : '•'} One special character (recommended)
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

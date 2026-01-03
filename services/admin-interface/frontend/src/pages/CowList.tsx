import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { cowsApi, CowIdentity } from '@/api/client'

interface SeverityStats {
  healthy: number
  mild: number
  moderate: number
  severe: number
  unknown: number
}

interface CowStats {
  total_cows: number
  active_cows: number
  total_videos_tracked: number
  total_lameness_records: number
  severity_distribution: SeverityStats
}

export default function CowList() {
  const [cows, setCows] = useState<CowIdentity[]>([])
  const [stats, setStats] = useState<CowStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>('')
  const [activeFilter, setActiveFilter] = useState<boolean | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Pagination
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)
  const limit = 20

  useEffect(() => {
    loadData()
  }, [severityFilter, activeFilter, skip])

  const loadData = async () => {
    try {
      setLoading(true)
      
      const [cowsData, statsData] = await Promise.all([
        cowsApi.list({
          skip,
          limit,
          is_active: activeFilter ?? undefined,
          severity_filter: severityFilter || undefined
        }),
        cowsApi.getStats()
      ])
      
      setCows(cowsData.cows)
      setTotal(cowsData.total)
      setStats(statsData)
      setError(null)
    } catch (err: any) {
      console.error('Failed to load cows:', err)
      setError(err.response?.data?.detail || 'Failed to load cow data')
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (severity: string | null | undefined): string => {
    switch (severity) {
      case 'healthy': return 'bg-emerald-100 text-emerald-800 border-emerald-200'
      case 'mild': return 'bg-amber-100 text-amber-800 border-amber-200'
      case 'moderate': return 'bg-orange-100 text-orange-800 border-orange-200'
      case 'severe': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-600 border-gray-200'
    }
  }

  const getSeverityIcon = (severity: string | null | undefined): string => {
    switch (severity) {
      case 'healthy': return '🐄'
      case 'mild': return '🟡'
      case 'moderate': return '🟠'
      case 'severe': return '🔴'
      default: return '❓'
    }
  }

  const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return 'Never'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  const filteredCows = cows.filter(cow => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return cow.cow_id.toLowerCase().includes(query) ||
           (cow.tag_number?.toLowerCase().includes(query) ?? false)
  })

  if (loading && cows.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading cow registry...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Cow Registry</h2>
          <p className="text-muted-foreground mt-1">
            Track and monitor individual cows across video analyses
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <StatCard 
            label="Total Cows" 
            value={stats.total_cows} 
            icon="🐮"
          />
          <StatCard 
            label="Active" 
            value={stats.active_cows} 
            icon="✅"
            className="bg-green-50 border-green-200"
          />
          <StatCard 
            label="Videos Tracked" 
            value={stats.total_videos_tracked} 
            icon="📹"
          />
          <StatCard 
            label="Healthy" 
            value={stats.severity_distribution.healthy} 
            className="bg-emerald-50 border-emerald-200"
            valueClassName="text-emerald-700"
          />
          <StatCard 
            label="Moderate" 
            value={stats.severity_distribution.moderate + stats.severity_distribution.mild} 
            className="bg-amber-50 border-amber-200"
            valueClassName="text-amber-700"
          />
          <StatCard 
            label="Severe" 
            value={stats.severity_distribution.severe} 
            className="bg-red-50 border-red-200"
            valueClassName="text-red-700"
          />
        </div>
      )}

      {/* Severity Distribution Chart */}
      {stats && (
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Severity Distribution</h3>
          <div className="flex items-center gap-2 h-8">
            {Object.entries(stats.severity_distribution).map(([severity, count]) => {
              const total = Object.values(stats.severity_distribution).reduce((a, b) => a + b, 0)
              const percentage = total > 0 ? (count / total) * 100 : 0
              if (percentage === 0) return null
              return (
                <div
                  key={severity}
                  className={`h-full rounded transition-all ${
                    severity === 'healthy' ? 'bg-emerald-500' :
                    severity === 'mild' ? 'bg-amber-400' :
                    severity === 'moderate' ? 'bg-orange-500' :
                    severity === 'severe' ? 'bg-red-500' : 'bg-gray-400'
                  }`}
                  style={{ width: `${percentage}%` }}
                  title={`${severity}: ${count} (${percentage.toFixed(1)}%)`}
                />
              )
            })}
          </div>
          <div className="flex gap-4 mt-3 text-sm">
            {Object.entries(stats.severity_distribution).map(([severity, count]) => (
              <div key={severity} className="flex items-center gap-1">
                <span className={`w-3 h-3 rounded-full ${
                  severity === 'healthy' ? 'bg-emerald-500' :
                  severity === 'mild' ? 'bg-amber-400' :
                  severity === 'moderate' ? 'bg-orange-500' :
                  severity === 'severe' ? 'bg-red-500' : 'bg-gray-400'
                }`} />
                <span className="capitalize">{severity}: {count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-4 items-center">
        <input
          type="text"
          placeholder="Search by cow ID or tag..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-4 py-2 border rounded-lg w-64 focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        
        <select
          value={severityFilter}
          onChange={(e) => setSeverityFilter(e.target.value)}
          className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">All Severities</option>
          <option value="healthy">Healthy</option>
          <option value="mild">Mild</option>
          <option value="moderate">Moderate</option>
          <option value="severe">Severe</option>
        </select>
        
        <select
          value={activeFilter === null ? '' : activeFilter ? 'active' : 'inactive'}
          onChange={(e) => {
            if (e.target.value === '') setActiveFilter(null)
            else setActiveFilter(e.target.value === 'active')
          }}
          className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">All Status</option>
          <option value="active">Active Only</option>
          <option value="inactive">Inactive Only</option>
        </select>
        
        <button
          onClick={loadData}
          className="px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      {/* Cow Table */}
      {filteredCows.length === 0 ? (
        <div className="text-center py-12 border rounded-lg bg-gray-50">
          <div className="text-4xl mb-4">🐮</div>
          <h4 className="text-lg font-semibold mb-2">No cows found</h4>
          <p className="text-muted-foreground">
            {cows.length === 0 
              ? 'Process videos with tracking enabled to identify cows'
              : 'No cows match your current filters'}
          </p>
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Cow ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Tag</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Severity</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Score</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Videos</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Last Seen</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filteredCows.map((cow) => (
                <tr key={cow.id} className="hover:bg-gray-50 transition-colors">
                  <td className="py-3 px-4">
                    <Link 
                      to={`/cows/${cow.cow_id}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {cow.cow_id.slice(0, 8)}...
                    </Link>
                  </td>
                  <td className="py-3 px-4">
                    {cow.tag_number ? (
                      <span className="px-2 py-1 bg-gray-100 rounded text-sm font-mono">
                        {cow.tag_number}
                      </span>
                    ) : (
                      <span className="text-muted-foreground text-sm">—</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(cow.severity_level)}`}>
                      {getSeverityIcon(cow.severity_level)}
                      <span className="capitalize">{cow.severity_level || 'Unknown'}</span>
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    {cow.current_score !== null && cow.current_score !== undefined ? (
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              cow.current_score < 0.3 ? 'bg-emerald-500' :
                              cow.current_score < 0.5 ? 'bg-amber-500' :
                              cow.current_score < 0.7 ? 'bg-orange-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${cow.current_score * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-mono">
                          {(cow.current_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-muted-foreground text-sm">—</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-sm">
                      {cow.num_videos ?? cow.total_sightings ?? 0}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm text-muted-foreground">
                    {formatDate(cow.last_seen)}
                  </td>
                  <td className="py-3 px-4">
                    {cow.is_active ? (
                      <span className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded-full">
                        Active
                      </span>
                    ) : (
                      <span className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
                        Inactive
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <Link
                      to={`/cows/${cow.cow_id}`}
                      className="px-3 py-1.5 text-xs bg-primary text-primary-foreground hover:bg-primary/90 rounded"
                    >
                      View Details →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {total > limit && (
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">
            Showing {skip + 1} - {Math.min(skip + limit, total)} of {total} cows
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setSkip(Math.max(0, skip - limit))}
              disabled={skip === 0}
              className="px-4 py-2 border rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              ← Previous
            </button>
            <button
              onClick={() => setSkip(skip + limit)}
              disabled={skip + limit >= total}
              className="px-4 py-2 border rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

interface StatCardProps {
  label: string
  value: number
  icon?: string
  className?: string
  valueClassName?: string
}

function StatCard({ label, value, icon, className = '', valueClassName = '' }: StatCardProps) {
  return (
    <div className={`p-4 border rounded-lg bg-background ${className}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon && <span>{icon}</span>}
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
      <p className={`text-2xl font-bold ${valueClassName}`}>{value}</p>
    </div>
  )
}


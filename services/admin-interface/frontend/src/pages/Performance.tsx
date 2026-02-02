import { useEffect, useState } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

export default function Performance() {
  const { t } = useLanguage()
  const [stats, setStats] = useState({
    totalRatings: 47,
    accuracy: 82.3,
    meanDeviation: 0.8,
    standardDeviation: 1.2,
    consistency: 88.5,
  })

  // Demo data for distribution chart
  const distributionData = [
    { deviation: '-3σ', count: 2, percentage: 2 },
    { deviation: '-2σ', count: 8, percentage: 8 },
    { deviation: '-1σ', count: 18, percentage: 18 },
    { deviation: 'μ', count: 25, percentage: 25, isYou: false },
    { deviation: t('performance.you'), count: 23, percentage: 23, isYou: true },
    { deviation: '+1σ', count: 16, percentage: 16 },
    { deviation: '+2σ', count: 6, percentage: 6 },
    { deviation: '+3σ', count: 2, percentage: 2 },
  ]

  const maxCount = Math.max(...distributionData.map(d => d.count))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold">{t('performance.title')}</h2>
        <p className="text-muted-foreground mt-1">
          {t('performance.subtitle')}
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.totalRatings')}</div>
          <div className="text-2xl font-bold mt-1">{stats.totalRatings}</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.accuracy')}</div>
          <div className="text-2xl font-bold mt-1 text-success">{stats.accuracy}%</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.meanDeviation')}</div>
          <div className="text-2xl font-bold mt-1">{stats.meanDeviation}σ</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.consistency')}</div>
          <div className="text-2xl font-bold mt-1 text-primary">{stats.consistency}%</div>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-2">{t('performance.distributionTitle')}</h3>
        <p className="text-sm text-muted-foreground mb-6">
          {t('performance.distributionDesc')} <span className="font-bold text-primary">{stats.standardDeviation}σ</span> {t('performance.distributionDescEnd')}
        </p>

        {/* Bell Curve Visualization */}
        <div className="space-y-2">
          {distributionData.map((item, idx) => (
            <div key={idx} className="flex items-center gap-3">
              {/* Label */}
              <div className="w-16 text-right text-sm font-medium">
                {item.deviation}
              </div>

              {/* Bar */}
              <div className="flex-1 h-12 relative">
                <div 
                  className={`h-full rounded-lg transition-all ${
                    item.isYou 
                      ? 'bg-gradient-to-r from-primary to-primary/70 border-2 border-primary shadow-lg shadow-primary/20' 
                      : 'bg-muted hover:bg-muted/80'
                  }`}
                  style={{ width: `${(item.count / maxCount) * 100}%` }}
                >
                  <div className="flex items-center justify-end h-full px-3">
                    <span className={`text-sm font-medium ${item.isYou ? 'text-primary-foreground' : 'text-muted-foreground'}`}>
                      {item.count}
                    </span>
                  </div>
                </div>
                {item.isYou && (
                  <div className="absolute -right-2 top-1/2 -translate-y-1/2">
                    <div className="bg-primary text-primary-foreground text-xs px-2 py-1 rounded-full font-bold shadow-lg">
                      {t('performance.you')}
                    </div>
                  </div>
                )}
              </div>

              {/* Percentage */}
              <div className="w-12 text-sm text-muted-foreground">
                {item.percentage}%
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-6 pt-4 border-t border-border">
          <div className="flex items-center justify-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-muted"></div>
              <span className="text-muted-foreground">{t('performance.otherRaters')}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-primary"></div>
              <span className="text-muted-foreground">{t('performance.yourPosition')}</span>
            </div>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-2">
            {t('performance.sigmaNote')}
          </p>
        </div>
      </div>

      {/* Performance Insights */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">{t('performance.insightsTitle')}</h3>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-3 bg-success/10 border border-success/30 rounded-lg">
            <div className="text-2xl">✓</div>
            <div>
              <div className="font-semibold text-success">{t('performance.goodConsistency')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.consistencyDesc')} {stats.standardDeviation}σ)
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-primary/10 border border-primary/30 rounded-lg">
            <div className="text-2xl">📊</div>
            <div>
              <div className="font-semibold text-primary">{t('performance.aboveAverage')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.aboveAverageDesc')} {100 - stats.accuracy}% {t('performance.aboveAverageDescEnd')}
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-muted/50 border border-border rounded-lg">
            <div className="text-2xl">💡</div>
            <div>
              <div className="font-semibold">{t('performance.keepRating')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.keepRatingDesc')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

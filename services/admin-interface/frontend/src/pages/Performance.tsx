import { useEffect, useState } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

export default function Performance() {
  const { t } = useLanguage()
  const [stats, setStats] = useState({
    totalRatings: 47,
    alignment: 85.4,  // Combined accuracy and consistency metric
    meanDeviation: 0.8,
    standardDeviation: 1.2,
  })

  const percentileOutperformed = 77 // User outperformed 77% of other raters

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold">Your Rating Performance</h2>
        <p className="text-muted-foreground mt-1">
          See how well your ratings match with other raters
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="text-sm text-muted-foreground mb-1">Total Ratings Completed</div>
          <div className="text-4xl font-bold mt-2">{stats.totalRatings}</div>
          <p className="text-xs text-muted-foreground mt-2">Number of cow comparisons you've rated</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="text-sm text-muted-foreground mb-1">Agreement Score</div>
          <div className="text-4xl font-bold mt-2 text-success">{stats.alignment}%</div>
          <p className="text-xs text-muted-foreground mt-2">How often your ratings match other raters</p>
        </div>
      </div>

      {/* Performance Ranking */}
      <div className="bg-card border border-border rounded-lg p-8">
        <div className="text-center mb-6">
          <div className="text-6xl font-bold text-success mb-3">{percentileOutperformed}%</div>
          <p className="text-xl font-medium mb-2">
            You're in the <span className="text-success">Top {100 - percentileOutperformed}%</span>
          </p>
          <p className="text-muted-foreground">
            You rate more accurately than {percentileOutperformed} out of 100 raters
          </p>
        </div>
        
        {/* Simple visual ranking bar */}
        <div className="relative w-full h-16 mt-8 bg-muted rounded-full overflow-hidden">
          {/* Progress fill */}
          <div 
            className="absolute top-0 left-0 h-full bg-gradient-to-r from-success/40 to-success/60 rounded-full"
            style={{ width: `${percentileOutperformed}%` }}
          />
          
          {/* User position marker */}
          <div 
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-success rounded-full border-4 border-white shadow-lg"
            style={{ left: `${percentileOutperformed}%`, transform: 'translate(-50%, -50%)' }}
          />
          
          {/* Labels */}
          <div className="absolute top-1/2 left-4 -translate-y-1/2 text-sm font-medium text-muted-foreground">
            0%
          </div>
          <div className="absolute top-1/2 right-4 -translate-y-1/2 text-sm font-medium text-muted-foreground">
            100%
          </div>
          <div 
            className="absolute -top-8 text-sm font-bold text-success"
            style={{ left: `${percentileOutperformed}%`, transform: 'translateX(-50%)' }}
          >
            You are here
          </div>
        </div>
      </div>

      {/* What This Means */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">What This Means</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-4 p-4 bg-success/10 border border-success/30 rounded-lg">
            <div className="text-3xl">✓</div>
            <div>
              <div className="font-semibold text-success mb-1">You're Doing Great!</div>
              <div className="text-sm text-muted-foreground">
                Your ratings are very consistent and match well with other experienced raters. Keep up the good work!
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div className="text-3xl">🎯</div>
            <div>
              <div className="font-semibold text-blue-500 mb-1">High Agreement Rate</div>
              <div className="text-sm text-muted-foreground">
                {stats.alignment}% of your ratings match the consensus. This shows you can reliably identify lameness in cows.
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4 p-4 bg-muted/50 border border-border rounded-lg">
            <div className="text-3xl">📈</div>
            <div>
              <div className="font-semibold mb-1">Keep Rating</div>
              <div className="text-sm text-muted-foreground">
                The more comparisons you complete, the more accurate the system becomes at detecting lameness patterns.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

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

  // User's best 5 answers with their deviations from group consensus
  // All have SD < 0.5 since these are the best answers (closest to group consensus)
  const bestAnswers = [
    { id: 1, title: 'Comparison #47', deviation: 0.08, skewness: 0.2 },   // Perfect alignment
    { id: 2, title: 'Comparison #32', deviation: 0.15, skewness: -0.3 },  // Excellent
    { id: 3, title: 'Comparison #18', deviation: 0.25, skewness: 0.4 },   // Very good
    { id: 4, title: 'Comparison #9', deviation: 0.35, skewness: -0.2 },   // Good
    { id: 5, title: 'Comparison #3', deviation: 0.45, skewness: 0.1 },    // Good
  ]

  const percentileOutperformed = 77 // User outperformed 77% of other raters

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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.totalRatings')}</div>
          <div className="text-2xl font-bold mt-1">{stats.totalRatings}</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Alignment</div>
          <div className="text-2xl font-bold mt-1 text-success">{stats.alignment}%</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">{t('performance.meanDeviation')}</div>
          <div className="text-2xl font-bold mt-1">{stats.meanDeviation}σ</div>
        </div>
      </div>

      {/* Performance Summary with Graph */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="text-center mb-6">
          <div className="text-5xl font-bold text-success mb-2">{percentileOutperformed}%</div>
          <p className="text-lg text-muted-foreground">
            You have outperformed <span className="font-semibold text-foreground">{percentileOutperformed}% of users</span>
          </p>
        </div>
        
        {/* Percentile visualization */}
        <div className="relative w-full h-48 mt-8">
          <svg className="w-full h-full" viewBox="0 0 600 150" preserveAspectRatio="xMidYMid meet">
            {/* Background gradient area (representing all users) */}
            <defs>
              <linearGradient id="populationGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="currentColor" stopOpacity="0.1" className="text-muted-foreground" />
                <stop offset="50%" stopColor="currentColor" stopOpacity="0.2" className="text-muted-foreground" />
                <stop offset="100%" stopColor="currentColor" stopOpacity="0.1" className="text-muted-foreground" />
              </linearGradient>
              <linearGradient id="userGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="currentColor" stopOpacity="0.3" className="text-success" />
                <stop offset="100%" stopColor="currentColor" stopOpacity="0.6" className="text-success" />
              </linearGradient>
            </defs>
            
            {/* Population distribution curve */}
            <path
              d="M 50,120 Q 150,20 300,20 T 550,120"
              fill="url(#populationGradient)"
              stroke="currentColor"
              strokeWidth="2"
              className="stroke-muted-foreground/50"
            />
            
            {/* User's position area (left side up to 77%) */}
            <clipPath id="userClip">
              <rect x="50" y="0" width={`${(percentileOutperformed / 100) * 500}`} height="150" />
            </clipPath>
            
            <path
              d="M 50,120 Q 150,20 300,20 T 550,120"
              fill="url(#userGradient)"
              stroke="currentColor"
              strokeWidth="2"
              className="stroke-success"
              clipPath="url(#userClip)"
            />
            
            {/* User position marker */}
            <line 
              x1={50 + (percentileOutperformed / 100) * 500}
              y1="25" 
              x2={50 + (percentileOutperformed / 100) * 500}
              y2="120" 
              stroke="currentColor" 
              strokeWidth="3" 
              className="stroke-success"
              strokeDasharray="5,5"
            />
            
            <circle 
              cx={50 + (percentileOutperformed / 100) * 500}
              cy="70" 
              r="6" 
              fill="currentColor" 
              className="fill-success" 
            />
            
            {/* Labels */}
            <text x="50" y="145" textAnchor="start" className="fill-muted-foreground text-xs">0%</text>
            <text x="300" y="145" textAnchor="middle" className="fill-muted-foreground text-xs">50%</text>
            <text x="550" y="145" textAnchor="end" className="fill-muted-foreground text-xs">100%</text>
            
            {/* User label */}
            <text 
              x={50 + (percentileOutperformed / 100) * 500}
              y="15" 
              textAnchor="middle" 
              className="fill-success font-bold text-sm"
            >
              You
            </text>
          </svg>
        </div>
        
        <p className="text-center text-xs text-muted-foreground mt-4">
          You perform better than {percentileOutperformed}% of all raters in the system
        </p>
      </div>

      {/* Best 5 Answers */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Your Best 5 Answers</h3>
        <p className="text-sm text-muted-foreground mb-6">
          These comparisons show your closest alignment with expert consensus
        </p>
        
        <div className="grid grid-cols-5 gap-4">
          {bestAnswers.map((answer) => {
            // User's position relative to group consensus (smaller deviation = better)
            const userOffset = answer.deviation * 60 // Scale deviation for positioning
            const userX = 60 + (userOffset * (answer.skewness > 0 ? 1 : -1))
            
            // Create unique curve shape based on skewness
            const skew = answer.skewness
            const peakX = 60 // Group consensus at center
            
            // Adjust control points for skewness to create asymmetric curves
            const leftHeight = skew < 0 ? 25 : 35
            const rightHeight = skew > 0 ? 25 : 35
            const peakY = 20 // Peak height
            
            return (
              <div key={answer.id} className="flex flex-col items-center">
                <div className="text-xs font-medium mb-2 text-muted-foreground">{answer.title}</div>
                
                {/* Mini distribution curve (unique shape for each) */}
                <svg className="w-full h-24" viewBox="0 0 120 120" preserveAspectRatio="xMidYMid meet">
                  {/* Smoother bell curve using cubic bezier */}
                  <path
                    d={`M 10,95 
                        C 20,${95 - leftHeight} 40,${peakY + 15} ${peakX},${peakY + 10}
                        C 80,${peakY + 15} 100,${95 - rightHeight} 110,95`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    className="stroke-primary/50"
                  />
                  
                  {/* Shaded area under curve */}
                  <path
                    d={`M 10,95 
                        C 20,${95 - leftHeight} 40,${peakY + 15} ${peakX},${peakY + 10}
                        C 80,${peakY + 15} 100,${95 - rightHeight} 110,95
                        L 110,95 L 10,95 Z`}
                    fill="currentColor"
                    className="fill-primary/10"
                  />
                  
                  {/* Group consensus line (at center peak) */}
                  <line 
                    x1={peakX}
                    y1={peakY + 10} 
                    x2={peakX}
                    y2="95" 
                    stroke="currentColor" 
                    strokeWidth="1" 
                    className="stroke-muted-foreground/30"
                    strokeDasharray="2,2"
                  />
                  
                  {/* User's position line */}
                  <line 
                    x1={userX}
                    y1={peakY + 15} 
                    x2={userX}
                    y2="95" 
                    stroke="currentColor" 
                    strokeWidth="2.5" 
                    className="stroke-success"
                  />
                  
                  {/* Dot at user's position */}
                  <circle 
                    cx={userX}
                    cy={peakY + 17}
                    r="3" 
                    fill="currentColor" 
                    className="fill-success" 
                  />
                  
                  {/* Labels */}
                  <text x={peakX} y="108" textAnchor="middle" className="fill-muted-foreground text-[8px]">
                    Group
                  </text>
                </svg>
                
                {/* Deviation label */}
                <div className="text-xs mt-2">
                  <span className={`font-mono ${answer.deviation < 0.2 ? 'text-emerald-500 font-bold' : answer.deviation < 0.35 ? 'text-success' : 'text-primary'}`}>
                    {answer.deviation.toFixed(2)}σ
                  </span>
                </div>
              </div>
            )
          })}
        </div>
        
        <div className="mt-6 pt-4 border-t border-border">
          <div className="flex items-center justify-center gap-6 text-xs mb-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-primary/30 border border-primary/50"></div>
              <span className="text-muted-foreground">Group Response Distribution</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-success"></div>
              <span className="text-muted-foreground">Your Response</span>
            </div>
          </div>
          <p className="text-xs text-center text-muted-foreground">
            Lower σ (standard deviation) values indicate closer alignment with group consensus. All displayed answers have σ &lt; 0.5, representing your best performance.
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
              <div className="font-semibold text-primary">Strong Alignment</div>
              <div className="text-sm text-muted-foreground">
                Your ratings show {stats.alignment}% alignment with expert consensus
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

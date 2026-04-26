import React from 'react';

export default function PriorityActions({ recommendations }) {
  if (!recommendations) return null;

  const { urgent_actions, monitor_actions } = recommendations;

  return (
    <div className="priorityActionsCard animate-slideUp delay-1">
      <h3>Priority Actions - What to Do Next</h3>
      {urgent_actions && urgent_actions.length > 0 ? (
        <div className="priorityAction urgent">
          <div className="priorityHeader">
            <span className="priorityIcon"></span>
            <strong>{urgent_actions[0].action}</strong>
          </div>
          <p>{urgent_actions[0].reason}</p>
          <span className="priorityTimeline">{urgent_actions[0].timeline}</span>
        </div>
      ) : monitor_actions && monitor_actions.length > 0 ? (
        <div className="priorityAction monitor">
          <div className="priorityHeader">
            <span className="priorityIcon"></span>
            <strong>{monitor_actions[0].action}</strong>
          </div>
          <p>{monitor_actions[0].reason}</p>
          <span className="priorityTimeline">{monitor_actions[0].timeline}</span>
        </div>
      ) : (
        <div className="priorityAction safe">
          <div className="priorityHeader">
            <span className="priorityIcon"></span>
            <strong>No immediate action required</strong>
          </div>
          <p>Continue regular monitoring and bias audits.</p>
        </div>
      )}
    </div>
  );
}

#!/usr/bin/env python3
"""
Agentic AI and Multi-Agent Systems Framework

Implementation of autonomous AI agents that can collaborate to perform complex tasks
independently, based on emerging 2025 trends in agentic AI systems.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import uuid

# Core libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# AI/ML libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Workflow orchestration
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    OFFLINE = "offline"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task definition for agent execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class Message:
    """Inter-agent communication message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.message_queue: List[Message] = []
        self.task_history: List[Task] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task"""
        pass
    
    @abstractmethod
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the given task"""
        pass
    
    async def send_message(self, recipient: str, message_type: str, 
                          content: Dict[str, Any], requires_response: bool = False):
        """Send message to another agent"""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            requires_response=requires_response
        )
        # In a real system, this would go through a message broker
        logger.info(f"Agent {self.name} sending message to {recipient}: {message_type}")
        return message
    
    async def receive_message(self, message: Message):
        """Receive and process a message"""
        self.message_queue.append(message)
        await self.process_message(message)
    
    async def process_message(self, message: Message):
        """Process received message"""
        logger.info(f"Agent {self.name} processing message: {message.message_type}")
        # Override in subclasses for specific message handling
        pass
    
    def update_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
        self.last_activity = datetime.now()

class LLMAgent(BaseAgent):
    """Language model-based agent for text processing tasks"""
    
    def __init__(self, agent_id: str, name: str, model_provider: str = "openai"):
        capabilities = [
            "text_generation", "text_analysis", "summarization",
            "translation", "question_answering", "content_creation"
        ]
        super().__init__(agent_id, name, capabilities)
        self.model_provider = model_provider
        self.setup_llm()
    
    def setup_llm(self):
        """Setup language model client"""
        if self.model_provider == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI()
            self.model = "gpt-4-turbo-preview"
        elif self.model_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
            self.model = "claude-3-sonnet-20240229"
        else:
            logger.warning(f"LLM provider {self.model_provider} not available")
            self.client = None
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the task"""
        return task.task_type in self.capabilities
    
    async def execute_task(self, task: Task) -> Any:
        """Execute LLM-based task"""
        self.update_status(AgentStatus.WORKING)
        
        try:
            if task.task_type == "text_generation":
                result = await self.generate_text(task.parameters)
            elif task.task_type == "text_analysis":
                result = await self.analyze_text(task.parameters)
            elif task.task_type == "summarization":
                result = await self.summarize_text(task.parameters)
            elif task.task_type == "question_answering":
                result = await self.answer_question(task.parameters)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = "completed"
            self.update_status(AgentStatus.COMPLETED)
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            self.update_status(AgentStatus.FAILED)
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def generate_text(self, parameters: Dict[str, Any]) -> str:
        """Generate text using LLM"""
        if not self.client:
            raise RuntimeError("LLM client not available")
        
        prompt = parameters.get("prompt", "")
        max_tokens = parameters.get("max_tokens", 1000)
        
        if self.model_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.model_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    async def analyze_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for sentiment, topics, etc."""
        text = parameters.get("text", "")
        analysis_type = parameters.get("analysis_type", "sentiment")
        
        prompt = f"""
        Analyze the following text for {analysis_type}:
        
        Text: {text}
        
        Provide a structured analysis including:
        - Main themes
        - Sentiment (positive/negative/neutral with confidence)
        - Key entities
        - Summary
        
        Return as JSON format.
        """
        
        result = await self.generate_text({"prompt": prompt, "max_tokens": 500})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_analysis": result}
    
    async def summarize_text(self, parameters: Dict[str, Any]) -> str:
        """Summarize given text"""
        text = parameters.get("text", "")
        length = parameters.get("length", "medium")
        
        prompt = f"""
        Summarize the following text in {length} length:
        
        {text}
        
        Provide a clear, concise summary that captures the main points.
        """
        
        return await self.generate_text({"prompt": prompt, "max_tokens": 300})
    
    async def answer_question(self, parameters: Dict[str, Any]) -> str:
        """Answer questions based on context"""
        question = parameters.get("question", "")
        context = parameters.get("context", "")
        
        prompt = f"""
        Context: {context}
        
        Question: {question}
        
        Provide a detailed answer based on the given context.
        """
        
        return await self.generate_text({"prompt": prompt, "max_tokens": 500})

class DataAgent(BaseAgent):
    """Agent specialized in data processing and analysis"""
    
    def __init__(self, agent_id: str, name: str):
        capabilities = [
            "data_analysis", "data_cleaning", "data_transformation",
            "statistical_analysis", "data_visualization", "data_validation"
        ]
        super().__init__(agent_id, name, capabilities)
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the task"""
        return task.task_type in self.capabilities
    
    async def execute_task(self, task: Task) -> Any:
        """Execute data processing task"""
        self.update_status(AgentStatus.WORKING)
        
        try:
            if task.task_type == "data_analysis":
                result = await self.analyze_data(task.parameters)
            elif task.task_type == "data_cleaning":
                result = await self.clean_data(task.parameters)
            elif task.task_type == "statistical_analysis":
                result = await self.statistical_analysis(task.parameters)
            elif task.task_type == "data_validation":
                result = await self.validate_data(task.parameters)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = "completed"
            self.update_status(AgentStatus.COMPLETED)
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            self.update_status(AgentStatus.FAILED)
            logger.error(f"Data task execution failed: {e}")
            raise
    
    async def analyze_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        data = parameters.get("data")
        analysis_type = parameters.get("analysis_type", "descriptive")
        
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if analysis_type == "descriptive":
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "describe": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).to_dict()
            }
        
        elif analysis_type == "correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "highly_correlated_pairs": self._find_high_correlations(correlation_matrix)
            }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated variable pairs"""
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })
        return high_corr_pairs
    
    async def clean_data(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Clean data based on specified rules"""
        data = parameters.get("data")
        cleaning_rules = parameters.get("rules", {})
        
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Handle missing values
        if cleaning_rules.get("handle_missing", True):
            strategy = cleaning_rules.get("missing_strategy", "drop")
            if strategy == "drop":
                df = df.dropna()
            elif strategy == "fill_mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "fill_median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove duplicates
        if cleaning_rules.get("remove_duplicates", True):
            df = df.drop_duplicates()
        
        # Handle outliers
        if cleaning_rules.get("handle_outliers", False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    async def statistical_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        data = parameters.get("data")
        test_type = parameters.get("test_type", "basic")
        
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        results = {}
        
        if test_type == "basic":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                results[col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis()
                }
        
        return results
    
    async def validate_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality"""
        data = parameters.get("data")
        validation_rules = parameters.get("rules", {})
        
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        validation_results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues": []
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                validation_results["issues"].append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(count),
                    "percentage": count / len(df) * 100
                })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["issues"].append({
                "type": "duplicates",
                "count": int(duplicate_count),
                "percentage": duplicate_count / len(df) * 100
            })
        
        return validation_results

class TaskOrchestrator:
    """Orchestrates task execution across multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.task_graph: Dict[str, List[str]] = {}  # Task dependencies
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def create_task(self, name: str, task_type: str, parameters: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   dependencies: List[str] = None) -> Task:
        """Create a new task"""
        task = Task(
            name=name,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            dependencies=dependencies or []
        )
        self.task_queue.append(task)
        logger.info(f"Created task: {task.name} ({task.id})")
        return task
    
    def find_capable_agents(self, task: Task) -> List[BaseAgent]:
        """Find agents capable of handling the task"""
        capable_agents = []
        for agent in self.agents.values():
            if agent.can_handle_task(task) and agent.status in [AgentStatus.IDLE, AgentStatus.COMPLETED]:
                capable_agents.append(agent)
        return capable_agents
    
    def assign_task(self, task: Task) -> Optional[BaseAgent]:
        """Assign task to the most suitable agent"""
        capable_agents = self.find_capable_agents(task)
        
        if not capable_agents:
            logger.warning(f"No capable agents found for task: {task.name}")
            return None
        
        # Simple assignment strategy: choose least busy agent
        selected_agent = min(capable_agents, 
                           key=lambda a: len([t for t in self.task_queue if t.assigned_agent == a.agent_id]))
        
        task.assigned_agent = selected_agent.agent_id
        logger.info(f"Assigned task {task.name} to agent {selected_agent.name}")
        return selected_agent
    
    def check_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {t.id for t in self.completed_tasks}
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a single task"""
        if not self.check_dependencies(task):
            logger.info(f"Task {task.name} waiting for dependencies")
            return None
        
        agent = self.assign_task(task)
        if not agent:
            task.status = "failed"
            task.error = "No capable agent available"
            self.failed_tasks.append(task)
            return None
        
        try:
            result = await agent.execute_task(task)
            self.completed_tasks.append(task)
            
            # Remove from queue
            if task in self.task_queue:
                self.task_queue.remove(task)
            
            logger.info(f"Task {task.name} completed successfully")
            return result
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.failed_tasks.append(task)
            
            # Remove from queue
            if task in self.task_queue:
                self.task_queue.remove(task)
            
            logger.error(f"Task {task.name} failed: {e}")
            return None
    
    async def run_orchestrator(self, max_concurrent_tasks: int = 5):
        """Run the task orchestrator"""
        logger.info("Starting task orchestrator")
        
        while self.task_queue or any(agent.status == AgentStatus.WORKING for agent in self.agents.values()):
            # Get ready tasks (dependencies satisfied)
            ready_tasks = [
                task for task in self.task_queue 
                if task.status == "pending" and self.check_dependencies(task)
            ]
            
            # Sort by priority
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Execute tasks concurrently
            active_tasks = ready_tasks[:max_concurrent_tasks]
            
            if active_tasks:
                await asyncio.gather(*[self.execute_task(task) for task in active_tasks])
            else:
                # No ready tasks, wait a bit
                await asyncio.sleep(1)
        
        logger.info("Task orchestrator completed")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "status": agent.status.value,
                    "capabilities": agent.capabilities,
                    "last_activity": agent.last_activity.isoformat()
                }
                for agent_id, agent in self.agents.items()
            },
            "tasks": {
                "total": len(self.task_queue) + len(self.completed_tasks) + len(self.failed_tasks),
                "pending": len(self.task_queue),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks)
            },
            "performance": {
                "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) * 100
                if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0
            }
        }

class MultiAgentWorkflow:
    """High-level workflow management for complex multi-step processes"""
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.workflows: Dict[str, Dict] = {}
    
    def define_workflow(self, workflow_id: str, workflow_definition: Dict[str, Any]):
        """Define a complex workflow with multiple steps"""
        self.workflows[workflow_id] = {
            "definition": workflow_definition,
            "created_at": datetime.now(),
            "status": "defined"
        }
    
    async def execute_data_analysis_workflow(self, data_source: str, analysis_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete data analysis workflow"""
        workflow_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Data validation
        validation_task = self.orchestrator.create_task(
            name="Data Validation",
            task_type="data_validation",
            parameters={
                "data": data_source,
                "rules": analysis_requirements.get("validation_rules", {})
            },
            priority=TaskPriority.HIGH
        )
        
        # Step 2: Data cleaning (depends on validation)
        cleaning_task = self.orchestrator.create_task(
            name="Data Cleaning",
            task_type="data_cleaning",
            parameters={
                "data": data_source,
                "rules": analysis_requirements.get("cleaning_rules", {})
            },
            dependencies=[validation_task.id],
            priority=TaskPriority.HIGH
        )
        
        # Step 3: Statistical analysis (depends on cleaning)
        analysis_task = self.orchestrator.create_task(
            name="Statistical Analysis",
            task_type="statistical_analysis",
            parameters={
                "data": data_source,
                "test_type": analysis_requirements.get("analysis_type", "basic")
            },
            dependencies=[cleaning_task.id],
            priority=TaskPriority.MEDIUM
        )
        
        # Step 4: Generate report (depends on analysis)
        report_task = self.orchestrator.create_task(
            name="Generate Report",
            task_type="text_generation",
            parameters={
                "prompt": f"""
                Generate a comprehensive data analysis report based on the following analysis results.
                Include key findings, recommendations, and visualizations suggestions.
                Analysis requirements: {analysis_requirements}
                """,
                "max_tokens": 2000
            },
            dependencies=[analysis_task.id],
            priority=TaskPriority.MEDIUM
        )
        
        # Execute workflow
        await self.orchestrator.run_orchestrator()
        
        return {
            "workflow_id": workflow_id,
            "tasks": [validation_task.id, cleaning_task.id, analysis_task.id, report_task.id],
            "status": "completed",
            "results": {
                "validation": validation_task.result,
                "cleaning": cleaning_task.result,
                "analysis": analysis_task.result,
                "report": report_task.result
            }
        }

# Example usage and demonstration
async def demo_agentic_ai_system():
    """Demonstrate the agentic AI system"""
    
    print("=== Agentic AI Multi-Agent System Demo ===\n")
    
    # Create orchestrator
    orchestrator = TaskOrchestrator()
    
    # Create and register agents
    llm_agent = LLMAgent("llm_001", "TextProcessor", "openai")
    data_agent = DataAgent("data_001", "DataAnalyzer")
    
    orchestrator.register_agent(llm_agent)
    orchestrator.register_agent(data_agent)
    
    # Create sample data for analysis
    sample_data = {
        "sales": [100, 120, 80, 150, 200, 90, 110],
        "marketing_spend": [10, 15, 8, 20, 25, 9, 12],
        "region": ["North", "South", "East", "West", "North", "South", "East"]
    }
    
    print("1. Creating individual tasks...")
    
    # Create tasks
    data_analysis_task = orchestrator.create_task(
        name="Analyze Sales Data",
        task_type="data_analysis",
        parameters={
            "data": sample_data,
            "analysis_type": "descriptive"
        },
        priority=TaskPriority.HIGH
    )
    
    text_summary_task = orchestrator.create_task(
        name="Generate Analysis Summary",
        task_type="text_generation",
        parameters={
            "prompt": "Generate a business summary of sales data analysis including key insights and recommendations.",
            "max_tokens": 500
        },
        dependencies=[data_analysis_task.id],
        priority=TaskPriority.MEDIUM
    )
    
    print("2. Executing tasks...")
    
    # Execute orchestrator
    await orchestrator.run_orchestrator()
    
    print("3. Results:")
    status_report = orchestrator.get_status_report()
    print(f"Success rate: {status_report['performance']['success_rate']:.1f}%")
    print(f"Completed tasks: {status_report['tasks']['completed']}")
    print(f"Failed tasks: {status_report['tasks']['failed']}")
    
    print("\n4. Workflow execution...")
    
    # Demonstrate workflow
    workflow_manager = MultiAgentWorkflow(orchestrator)
    
    # Reset for workflow demo
    orchestrator.completed_tasks = []
    orchestrator.failed_tasks = []
    
    workflow_result = await workflow_manager.execute_data_analysis_workflow(
        data_source=sample_data,
        analysis_requirements={
            "analysis_type": "correlation",
            "cleaning_rules": {"handle_missing": True, "remove_duplicates": True},
            "validation_rules": {}
        }
    )
    
    print(f"Workflow {workflow_result['workflow_id']} completed with {len(workflow_result['tasks'])} tasks")
    
    return orchestrator, workflow_manager

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_agentic_ai_system()) 
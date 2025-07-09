import asyncio
import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - just use system environment variables
    pass


class StudyType(str, Enum):
    """Type of study described in the article."""
    DESCRIPTIVE = "descriptive"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    UNCLEAR = "unclear"


class RecommendationEntry(str, Enum):
    """How recommendations are entered into tracking systems."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    BOTH = "both"
    NOT_SPECIFIED = "not_specified"


class BasicAnalysis(BaseModel):
    """Basic yes/no analysis of the article's key features."""
    
    # Question 1
    study_type: StudyType = Field(
        description="Whether the article was descriptive or tested hypotheses"
    )
    
    # Questions 2-8 (basic yes/no)
    describes_identification_method: bool = Field(
        description="Does the article describe a method for identifying follow-up recommendations in radiology reports?"
    )
    describes_communication_beyond_report: bool = Field(
        description="Does the article describe how follow-up recommendations were communicated beyond the radiology report?"
    )
    describes_tracking_system: bool = Field(
        description="Does the article describe follow-up recommendations being entered into a tracking system?"
    )
    describes_completion_determination: bool = Field(
        description="Does the article describe how the tracking system determined if a recommendation was completed?"
    )
    describes_ordering_assistance: bool = Field(
        description="Does the article describe assistance with ordering/scheduling of recommended follow-up exams?"
    )
    describes_outcome_tracking: bool = Field(
        description="Does the article describe tracking of outcomes associated with exams performed due to recommendations?"
    )
    describes_influencing_factors: bool = Field(
        description="Does the article describe patient or system factors that influence recommendation completion?"
    )
    
    # Confidence
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for the basic analysis (0-1)"
    )


# Individual detail prompts
HYPOTHESIS_PROMPT = """You previously identified that this article tests at least one hypothesis. 

What was the specific hypothesis tested in this study? Provide the exact hypothesis or describe it as clearly as possible based on the article text in 1-2 sentences."""

class HypothesisDetails(BaseModel):
    """Details about the hypothesis tested in the study."""
    hypothesis: str = Field(description="The specific hypothesis tested (1-2 sentences)")

IDENTIFICATION_METHOD_PROMPT = """You previously identified that this article describes a method for identifying follow-up recommendations in radiology reports.

Please provide specific details about the identification method by answering these questions:

a. How are the recommendations identified? For example, were the recommendations identified using NLP or AI or an LLM or were they manually flagged by radiologists?

b. Did this article focus on a particular type of radiology finding and follow up recommendation (such as a pulmonary nodule and recommendation for follow up imaging of the pulmonary nodule), if so, what?

c. Does the article describe how well the method for identifying the follow up recommendation worked (such as sensitivity, specificity, precision, recall, accuracy, and f-score)? If so, describe the performance of the method in identifying follow up recommendations."""

class IdentificationMethodDetails(BaseModel):
    """Details about how follow-up recommendations are identified."""
    identification_approach: str = Field(description="How recommendations are identified (NLP, AI, LLM, manual flagging by radiologists, etc.)")
    specific_finding_focus: Optional[str] = Field(default=None, description="Specific type of radiology finding focused on (e.g., pulmonary nodule, liver lesion), if any")
    performance_described: bool = Field(description="Whether the article describes performance metrics for the identification method")
    performance_metrics: Optional[str] = Field(default=None, description="Performance metrics reported (sensitivity, specificity, precision, recall, accuracy, f-score, etc.) if described")

COMMUNICATION_PROMPT = """You previously identified that this article describes how follow-up recommendations were communicated beyond the radiology report.

Please provide specific details about the communication of follow-up recommendations by answering these questions:

a. Who received communication about the follow up recommendation (for example, the patient, referring provider, primary care provider, or a specialist)?

b. How was the recommendation communicated (for example, via telephone, email, message through the electronic health record)?

c. When did the communication occur (around the time the recommendation was made in the report, shortly before the recommended follow up was due, or once the recommended follow up was overdue)?"""

class CommunicationDetails(BaseModel):
    """Details about how recommendations are communicated beyond reports."""
    communication_recipients: str = Field(description="Who received communication about the follow-up recommendation (patient, referring provider, primary care provider, specialist, etc.)")
    communication_methods: str = Field(description="How the recommendation was communicated (telephone, email, electronic health record message, etc.)")
    communication_timing: str = Field(description="When the communication occurred (at time of report, before follow-up due, after overdue, etc.)")

TRACKING_SYSTEM_PROMPT = """You previously identified that this article describes follow-up recommendations being entered into a tracking system.

Please provide details about:
1. Were the recommendations entered automatically or manually?
2. What are the specific details about the tracking system described?"""


class TrackingSystemDetails(BaseModel):
    """Details about the tracking system for recommendations."""
    entry_method: RecommendationEntry = Field(description="How recommendations are entered (automatic/manual)")
    system_description: str = Field(description="Description of the tracking system")

COMPLETION_DETERMINATION_PROMPT = """You previously identified that this article describes how the tracking system determined if a recommendation was completed.

Please provide specific details about completion determination by answering these questions:

a. How were overdue follow up recommendations monitored (by a human or a computer system)?

b. What action was taken if the patient was overdue for follow up?"""

class CompletionDeterminationDetails(BaseModel):
    """Details about how completion is determined."""
    monitoring_method: str = Field(description="How overdue follow-up recommendations were monitored (human or computer system)")
    overdue_actions: str = Field(description="What action was taken if the patient was overdue for follow up")


ORDERING_ASSISTANCE_PROMPT = """You previously identified that this article describes assistance with ordering/scheduling of recommended follow-up exams.

Please provide specific details about ordering assistance by answering these questions:

a. Describe how the system assists with ordering and/or scheduling of the follow up.

b. Is the ordering and/or scheduling of the follow up exam automated or done by humans?"""

class OrderingAssistanceDetails(BaseModel):
    """Details about assistance with ordering/scheduling."""
    assistance_description: str = Field(description="How the system assists with ordering and/or scheduling of the follow up")
    automation_level: str = Field(description="Whether the ordering and/or scheduling of the follow up exam is automated or done by humans")

OUTCOME_TRACKING_PROMPT = """You previously identified that this article describes tracking of outcomes associated with exams performed as a result of recommendations.

What specific outcomes were tracked for the exams performed as a result of the recommendations? List all outcomes mentioned."""

class OutcomeTrackingDetails(BaseModel):
    """Details about outcome tracking."""
    tracked_outcomes: List[str] = Field(description="List of specific outcomes that were tracked")

INFLUENCING_FACTORS_PROMPT = """You previously identified that this article describes patient or system factors that influence recommendation completion.

Please provide specific details about influencing factors by answering these questions:

a. Describe the patient and/or system factors associated with follow up recommendations being completed and/or not being completed.

b. For each factor, if a metric was given describing the extent to which follow up occurred, please provide the metric."""

class InfluencingFactorsDetails(BaseModel):
    """Details about factors that influence completion."""
    factors_description: str = Field(description="Patient and/or system factors associated with follow up recommendations being completed and/or not being completed")
    factor_metrics: Optional[str] = Field(default=None, description="Metrics describing the extent to which follow up occurred for each factor, if provided")


class ArticleAnalysis(BaseModel):
    """Complete structured analysis of a radiology follow-up recommendation article."""
    
    # Basic findings (always present)
    study_type: StudyType = Field(description="Whether the article was descriptive or tested hypotheses")
    describes_identification_method: bool = Field(description="Does the article describe a method for identifying follow-up recommendations?")
    describes_communication_beyond_report: bool = Field(description="Does the article describe communication beyond reports?")
    describes_tracking_system: bool = Field(description="Does the article describe a tracking system?")
    describes_completion_determination: bool = Field(description="Does the article describe completion determination?")
    describes_ordering_assistance: bool = Field(description="Does the article describe ordering assistance?")
    describes_outcome_tracking: bool = Field(description="Does the article describe outcome tracking?")
    describes_influencing_factors: bool = Field(description="Does the article describe influencing factors?")
    
    # Detailed findings (only present if relevant)
    hypothesis: Optional[str] = Field(default=None, description="The hypothesis tested, if applicable")
    identification_approach: Optional[str] = Field(default=None, description="How recommendations are identified")
    specific_finding_focus: Optional[str] = Field(default=None, description="Specific type of radiology finding focused on")
    performance_described: Optional[bool] = Field(default=None, description="Whether performance metrics are described")
    performance_metrics: Optional[str] = Field(default=None, description="Performance metrics reported")
    communication_recipients: Optional[str] = Field(default=None, description="Who received communication about follow-up recommendations")
    communication_methods: Optional[str] = Field(default=None, description="How recommendations were communicated")
    communication_timing: Optional[str] = Field(default=None, description="When communication occurred")
    tracking_entry_method: Optional[RecommendationEntry] = Field(default=None, description="How recommendations are entered")
    tracking_system_details: Optional[str] = Field(default=None, description="Details about the tracking system")
    completion_monitoring_method: Optional[str] = Field(default=None, description="How overdue follow-up recommendations were monitored")
    completion_overdue_actions: Optional[str] = Field(default=None, description="Actions taken if patient was overdue for follow up")
    ordering_assistance_description: Optional[str] = Field(default=None, description="How the system assists with ordering and/or scheduling of follow up")
    ordering_automation_level: Optional[str] = Field(default=None, description="Whether ordering/scheduling is automated or done by humans")
    tracked_outcomes: List[str] = Field(default_factory=list, description="List of tracked outcomes")
    influencing_factors_description: Optional[str] = Field(default=None, description="Patient and/or system factors associated with follow up completion")
    influencing_factor_metrics: Optional[str] = Field(default=None, description="Metrics describing the extent to which follow up occurred for each factor")
    
    # Metadata
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class AnalysisResult(BaseModel):
    """Complete analysis result with usage tracking."""
    analysis: ArticleAnalysis = Field(description="The structured analysis results")
    total_usage: dict = Field(description="Combined usage statistics from all LLM calls")


SYSTEM_PROMPT = """You are an expert in analyzing scientific articles about radiology operations in clinical medicine. 

Your task is to extract key information from the provided article text about systems for managing recommendations for follow-up exams based on radiological examinations.

Analyze the article carefully and provide structured responses to specific questions about the study design and the system it describes. Be precise and evidence-based in your responses.

If information is not clearly stated in the article, indicate this rather than making assumptions."""

BASIC_ANALYSIS_PROMPT = """Analyze this article and answer the following basic questions with yes/no responses:

1. Was this article descriptive or did it test at least one hypothesis?

2. Does this article describe a method for identifying follow-up recommendations in radiology reports?

3. Does this article describe how the existence of follow-up recommendations was communicated beyond its inclusion in the radiology report?

4. Does this article describe follow-up recommendations being entered into a tracking system to ensure that they are completed?

5. Does this article describe how the tracking system determined if a recommendation was completed?

6. Does this article describe any assistance with the ordering/scheduling of recommended follow-up exams?

7. Does the article describe any tracking of the outcomes associated with the exams performed as a result of the recommendations?

8. Does the article describe patient or system factors that influence if a recommendation is completed or not?

Provide a confidence score based on how clearly the article addresses these questions."""

# Individual detail prompts
HYPOTHESIS_PROMPT = """You previously identified that this article tests at least one hypothesis. 

What was the specific hypothesis tested in this study? Provide the exact hypothesis or describe it as clearly as possible based on the article text in 1-2 sentences."""

IDENTIFICATION_METHOD_PROMPT = """You previously identified that this article describes a method for identifying follow-up recommendations in radiology reports.

What are the specific details about how follow-up recommendations were identified? Describe the method, technology, or process used."""

COMMUNICATION_PROMPT = """You previously identified that this article describes how follow-up recommendations were communicated beyond the radiology report.

How specifically were follow-up recommendations communicated beyond the radiology report? Describe the communication methods, systems, or processes used."""

TRACKING_SYSTEM_PROMPT = """You previously identified that this article describes follow-up recommendations being entered into a tracking system.

Please provide details about:
1. Were the recommendations entered automatically or manually?
2. What are the specific details about the tracking system described?"""

ORDERING_ASSISTANCE_PROMPT = """You previously identified that this article describes assistance with ordering/scheduling of recommended follow-up exams.

What specific assistance was provided for ordering/scheduling recommended follow-up exams? Describe the support, systems, or processes mentioned."""

OUTCOME_TRACKING_PROMPT = """You previously identified that this article describes tracking of outcomes associated with exams performed as a result of recommendations.

What specific outcomes were tracked for the exams performed as a result of the recommendations? List all outcomes mentioned."""


def check_api_configuration() -> bool:
    """
    Check if the OpenAI API is properly configured.
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        return False
    return True


def get_configuration_help() -> str:
    """Get help text for configuring the API."""
    return """
To use this analysis tool, you need to configure your OpenAI API key:

1. Get an API key from https://platform.openai.com/api-keys
2. Either:
   a. Set environment variable: export OPENAI_API_KEY="your-key-here"
   b. Create a .env file in this directory with: OPENAI_API_KEY=your-key-here

Example .env file:
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=openai:gpt-4o

You can copy .env.example to .env and edit it with your key.
"""


def get_article_analyzer():
    """Get the article analyzer agent, initializing it on first use."""
    if not check_api_configuration():
        raise Exception(
            "OpenAI API key not configured.\n" + get_configuration_help()
        )
    
    # Get model from environment or use default
    model_name = os.getenv("OPENAI_MODEL", "openai:gpt-4o")

    return Agent[None, ArticleAnalysis](
        model=model_name,
        output_type=ArticleAnalysis,
        system_prompt=SYSTEM_PROMPT,
    )


async def analyze_article(article_text: str) -> ArticleAnalysis:
    """
    Analyze a scientific article about radiology follow-up recommendations using a single conversation.
    
    Args:
        article_text: The full text of the article in markdown format
        
    Returns:
        ArticleAnalysis: Structured analysis results
        
    Raises:
        Exception: If OpenAI API key is not configured or analysis fails
    """
    try:
        if not check_api_configuration():
            raise Exception("OpenAI API key not configured.\n" + get_configuration_help())
        
        model_name = os.getenv("OPENAI_MODEL", "openai:gpt-4o")
        
        # Create a single agent that can handle multiple response types
        agent = Agent(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Create a single usage object to track across all calls
        total_usage = Usage()
        
        # Track message history throughout the conversation
        message_history: list[ModelMessage] | None = None
        
        # Initialize conversation with the article
        conversation = await agent.run(
            f"""I will analyze this scientific article about radiology follow-up recommendations in stages. 
            
            First, please analyze the article and answer these basic questions:

            {BASIC_ANALYSIS_PROMPT}

            Article text:
            {article_text}""",
            output_type=BasicAnalysis,
            usage=total_usage
        )
        
        basic_analysis = conversation.output
        message_history = conversation.all_messages()
        
        # Initialize the final analysis with basic results
        final_analysis = ArticleAnalysis(
            study_type=basic_analysis.study_type,
            describes_identification_method=basic_analysis.describes_identification_method,
            describes_communication_beyond_report=basic_analysis.describes_communication_beyond_report,
            describes_tracking_system=basic_analysis.describes_tracking_system,
            describes_completion_determination=basic_analysis.describes_completion_determination,
            describes_ordering_assistance=basic_analysis.describes_ordering_assistance,
            describes_outcome_tracking=basic_analysis.describes_outcome_tracking,
            describes_influencing_factors=basic_analysis.describes_influencing_factors,
            confidence_score=basic_analysis.confidence_score,
        )
        
        # Continue the conversation for detailed questions (only if relevant)
        
        # Get hypothesis details if it's a hypothesis-testing study
        if basic_analysis.study_type == StudyType.HYPOTHESIS_TESTING:
            hypothesis_result = await agent.run(
                HYPOTHESIS_PROMPT,
                output_type=HypothesisDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.hypothesis = hypothesis_result.output.hypothesis
            message_history = hypothesis_result.all_messages()
        
        # Get identification method details
        if basic_analysis.describes_identification_method:
            identification_result = await agent.run(
                IDENTIFICATION_METHOD_PROMPT,
                output_type=IdentificationMethodDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.identification_approach = identification_result.output.identification_approach
            final_analysis.specific_finding_focus = identification_result.output.specific_finding_focus
            final_analysis.performance_described = identification_result.output.performance_described
            final_analysis.performance_metrics = identification_result.output.performance_metrics
            message_history = identification_result.all_messages()
        
        # Get communication details
        if basic_analysis.describes_communication_beyond_report:
            communication_result = await agent.run(
                COMMUNICATION_PROMPT,
                output_type=CommunicationDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.communication_recipients = communication_result.output.communication_recipients
            final_analysis.communication_methods = communication_result.output.communication_methods
            final_analysis.communication_timing = communication_result.output.communication_timing
            message_history = communication_result.all_messages()
        
        # Get tracking system details
        if basic_analysis.describes_tracking_system:
            tracking_result = await agent.run(
                TRACKING_SYSTEM_PROMPT,
                output_type=TrackingSystemDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.tracking_entry_method = tracking_result.output.entry_method
            final_analysis.tracking_system_details = tracking_result.output.system_description
            message_history = tracking_result.all_messages()
        
        # Get completion determination details
        if basic_analysis.describes_completion_determination:
            completion_result = await agent.run(
                COMPLETION_DETERMINATION_PROMPT,
                output_type=CompletionDeterminationDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.completion_monitoring_method = completion_result.output.monitoring_method
            final_analysis.completion_overdue_actions = completion_result.output.overdue_actions
            message_history = completion_result.all_messages()
        
        # Get ordering assistance details
        if basic_analysis.describes_ordering_assistance:
            assistance_result = await agent.run(
                ORDERING_ASSISTANCE_PROMPT,
                output_type=OrderingAssistanceDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.ordering_assistance_description = assistance_result.output.assistance_description
            final_analysis.ordering_automation_level = assistance_result.output.automation_level
            message_history = assistance_result.all_messages()
        
        # Get outcome tracking details
        if basic_analysis.describes_outcome_tracking:
            outcome_result = await agent.run(
                OUTCOME_TRACKING_PROMPT,
                output_type=OutcomeTrackingDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.tracked_outcomes = outcome_result.output.tracked_outcomes
            message_history = outcome_result.all_messages()
        
        # Get influencing factors details
        if basic_analysis.describes_influencing_factors:
            factors_result = await agent.run(
                INFLUENCING_FACTORS_PROMPT,
                output_type=InfluencingFactorsDetails,
                message_history=message_history,
                usage=total_usage
            )
            final_analysis.influencing_factors_description = factors_result.output.factors_description
            final_analysis.influencing_factor_metrics = factors_result.output.factor_metrics
            message_history = factors_result.all_messages()
        
        # Store the total usage in the final analysis notes for now
        if total_usage:
            usage_summary = f"Total tokens: {getattr(total_usage, 'total_tokens', 'N/A')}"
            final_analysis.notes = f"Usage: {usage_summary}" + (f" | {final_analysis.notes}" if final_analysis.notes else "")
        
        return final_analysis
        
    except Exception as e:
        if "api_key" in str(e).lower():
            raise Exception(
                "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable "
                "or create a .env file with your API key."
            ) from e
        raise


async def analyze_article_from_file(file_path: str) -> ArticleAnalysis:
    """
    Analyze a scientific article from a markdown file.
    
    Args:
        file_path: Path to the markdown file containing the article
        
    Returns:
        ArticleAnalysis: Structured analysis results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        article_text = f.read()
    
    return await analyze_article(article_text)

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 20))

async def batch_analyze_articles(file_paths: List[str]) -> List[tuple[str, ArticleAnalysis]]:
    """
    Analyze multiple articles in batch.
    
    Args:
        file_paths: List of paths to markdown files containing articles
        
    Returns:
        List of tuples containing (file_path, analysis_result)
    """
    async def analyze_single(file_path: str, semaphore: asyncio.Semaphore) -> tuple[str, ArticleAnalysis]:
        async with semaphore:
            try:
                analysis = await analyze_article_from_file(file_path)
                return (file_path, analysis)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
            raise
    
    # Process articles concurrently (be mindful of API rate limits)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_single(path, semaphore) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return results


def export_analysis_to_dict(analysis: ArticleAnalysis) -> dict:
    """Convert analysis result to a dictionary for easy serialization."""
    return analysis.model_dump()


def save_batch_results_to_json(results: List[tuple[str, ArticleAnalysis]], output_path: str) -> None:
    """Save batch analysis results to a JSON file."""
    import json
    
    output_data = []
    for file_path, analysis in results:
        output_data.append({
            "file_path": file_path,
            "analysis": export_analysis_to_dict(analysis)
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def print_analysis_summary(analysis: ArticleAnalysis) -> None:
    """Print a human-readable summary of the analysis results."""
    print("Article Analysis Summary")
    print("=" * 50)
    print(f"Study Type: {analysis.study_type.value}")
    if analysis.hypothesis:
        print(f"Hypothesis: {analysis.hypothesis}")
    
    print("\nKey Findings:")
    print(f"  Describes identification method: {analysis.describes_identification_method}")
    if analysis.identification_approach:
        print(f"    Approach: {analysis.identification_approach}")
    if analysis.specific_finding_focus:
        print(f"    Focus: {analysis.specific_finding_focus}")
    if analysis.performance_described:
        print(f"    Performance described: {analysis.performance_described}")
    if analysis.performance_metrics:
        print(f"    Metrics: {analysis.performance_metrics}")
    
    print(f"  Describes communication beyond report: {analysis.describes_communication_beyond_report}")
    if analysis.communication_recipients:
        print(f"    Recipients: {analysis.communication_recipients}")
    if analysis.communication_methods:
        print(f"    Methods: {analysis.communication_methods}")
    if analysis.communication_timing:
        print(f"    Timing: {analysis.communication_timing}")
    print(f"  Describes tracking system: {analysis.describes_tracking_system}")
    if analysis.tracking_entry_method:
        print(f"    Entry method: {analysis.tracking_entry_method.value}")
    print(f"  Describes completion determination: {analysis.describes_completion_determination}")
    if analysis.completion_monitoring_method:
        print(f"    Monitoring method: {analysis.completion_monitoring_method}")
    if analysis.completion_overdue_actions:
        print(f"    Overdue actions: {analysis.completion_overdue_actions}")
    print(f"  Describes ordering assistance: {analysis.describes_ordering_assistance}")
    if analysis.ordering_assistance_description:
        print(f"    Assistance description: {analysis.ordering_assistance_description}")
    if analysis.ordering_automation_level:
        print(f"    Automation level: {analysis.ordering_automation_level}")
    print(f"  Describes outcome tracking: {analysis.describes_outcome_tracking}")
    if analysis.tracked_outcomes:
        print(f"    Tracked outcomes: {', '.join(analysis.tracked_outcomes)}")
    print(f"  Describes influencing factors: {analysis.describes_influencing_factors}")
    if analysis.influencing_factors_description:
        print(f"    Factors: {analysis.influencing_factors_description}")
    if analysis.influencing_factor_metrics:
        print(f"    Metrics: {analysis.influencing_factor_metrics}")
    
    print(f"\nConfidence Score: {analysis.confidence_score:.2f}")
    if analysis.notes:
        print(f"Notes: {analysis.notes}")


if __name__ == "__main__":
    import sys
    import logfire

    logfire.configure()
    logfire.instrument_pydantic_ai()


    async def analyze_single(filename: str):
        # Example: analyze a markdown file from the data/md directory
        try:
            analysis = await analyze_article_from_file(filename)
            print(analysis.model_dump_json(indent=2, exclude_none=True))
            
        except FileNotFoundError:
            print(f"File {filename} not found. Check the path.")
        except Exception as e:
            print(f"Error: {e}")

    async def analyze_multiple(files: List[str]):
        """Analyze multiple files concurrently."""
        try:
            results = await batch_analyze_articles(files)
            for file_path, analysis in results:
                print(f"\n========== 📄 Analysis for {file_path} ==========")
                print(analysis.model_dump_json(indent=2, exclude_none=True))
        except Exception as e:
            print(f"Error during batch analysis: {e}")
    
    if len(sys.argv) < 2:
        print("Usage: python article_analysis.py <path_to_markdown_file>")
    elif len(sys.argv) == 2:
        asyncio.run(analyze_single(sys.argv[1]))
    else:
        asyncio.run(analyze_multiple(sys.argv[1:]))




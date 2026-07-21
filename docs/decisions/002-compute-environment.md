# ADR-002: Compute Environment Strategy

## Status
Accepted

## Context

As the dataset size is closer to 2.5 GB, downloading and uploading of data requires a lot of time due to low internet bandwidth/speeds.

For that purpose, the plan is to develop the code and configuration locally and push the code to GitHub. The repository will then be pulled in Google Colab, where the environment will be set up and the code will be executed.

After the required data or artifacts are generated, they will be added to DVC and pushed to remote storage. The generated DVC files will then be pushed to GitHub through Colab or downloaded locally and pushed from the local environment.

## Decision

Google Colab will be used as the primary compute environment for executing heavy processing tasks.

The workflow will be:

Local machine:
- Development of code and configuration.

GitHub:
- Storage of source code and project files.

Google Colab:
- Execution of pipeline stages and computationally intensive tasks.

DVC remote:
- Storage of generated datasets and artifacts.

A setup script will be created to reduce the repeated environment setup process in Colab, including environment variables and authentication steps.

## Alternatives Considered

### Local Machine

The option was to use the local machine completely for development and execution.

It was rejected because of multiple obstacles:
- Dataset download/upload speeds due to limited internet bandwidth.
- Computational limitations for large-scale processing and training.

### Cloud Compute Services (AWS, Azure, etc.)

Cloud compute services were considered as an alternative.

They were rejected initially because:
- They are paid services.
- Initial setup and account configuration add additional complexity.

They may be considered later depending on project requirements.

### Google Colab Free Version

Google Colab free version was selected because:

Benefits:
- Free computation resources.
- Faster downloads/uploads compared to the local workflow.
- No requirement for managing cloud infrastructure.

Limitations:
- Environment setup needs to be repeated.
- Environment variables and authentication need to be configured.
- Session limitations can interrupt long-running tasks.

To minimize environment setup time, a setup script is created.

## Benefits

- Faster development and experimentation cycle.
- Reduced dependency on local hardware limitations.
- Easier execution of large processing tasks.
- Separation between development, computation, and artifact storage.

## Limitations

- Colab session limitations.
- Environment needs to be recreated after session expiry.
- Dependence on external compute availability.

## Future Considerations

In the future, the project might shift to paid Colab versions or cloud compute services to handle longer sessions and increased computational requirements.
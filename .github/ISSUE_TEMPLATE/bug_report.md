name: Bug Report
description: File a bug report for VulkanIlm
title: "[BUG]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: dropdown
    id: gpu
    attributes:
      label: What GPU are you using?
      multiple: false
      options:
        - AMD RX 580
        - AMD RX 6600/6700
        - Intel Arc A770/A750
        - NVIDIA GTX/RTX
        - Other AMD GPU
        - Other Intel GPU
        - Other NVIDIA GPU
    validations:
      required: true
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe the bug and what you expected to happen
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Run command...
        2. See error...
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output
      render: shell

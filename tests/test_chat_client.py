from finetuner.inference.chat_client import inference_metrics, reply_text


def test_reply_text_reads_chat_message():
    assert reply_text({"choices": [{"message": {"role": "assistant", "content": "hello"}}]}) == "hello"


def test_reply_text_reads_text_and_content_parts():
    assert reply_text({"choices": [{"text": "plain"}]}) == "plain"
    assert (
        reply_text({"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]})
        == "ab"
    )


def test_inference_metrics_include_speed_latency_and_tokens():
    text = inference_metrics(
        {
            "usage": {"prompt_tokens": 12, "completion_tokens": 30},
            "timings": {"prompt_ms": 40, "predicted_ms": 500, "predicted_per_second": 60},
        },
        0.82,
    )
    assert "60.0 tok/s" in text
    assert "820 ms" in text
    assert "prefill 40 ms" in text
    assert "12→30 tok" in text

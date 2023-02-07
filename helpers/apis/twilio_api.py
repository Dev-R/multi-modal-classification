"""
     Twilio API service
"""
from twilio.rest import Client
from ..constants import Config

# import geocoder
import time


def make_call(client, template_url):
    call = client.calls.create(
        url=template_url,
        to=Config.TO_PHONE_NUMBER,
        from_=Config.FROM_PHONE_NUMBER,
    )

    print(call.sid)


def send_sms(client: object, message_template: callable, message_data: dict) -> None:
    print(message_template(message_data))
    message = client.messages.create(
        body=message_template(message_data),
        from_=Config.FROM_PHONE_NUMBER,
        to=Config.TO_PHONE_NUMBER,
        messaging_service_sid=Config.MESSAGE_SERVICE_ID,
    )

    return message.sid


def simple_message_template_a(data):
    return f""" 
        An anomaly alert! ⚠
        Time: { data.get('time', 'N/A') }
        Location: { data.get('location', 'N/A') }
        Anomaly Audio classification: { data.get('audio_classification', 'N/A')}
        Anomaly Video classification: { data.get('video_classification', 'N./A') }
        Check evidence here:
        Image: { data.get('image_link', 'N/A') }
        Video: { data.get('video_link', 'N/A') }
        Please take necessary actions and contact us if you have any questions or concerns. 
    """


def simple_message_template_b(data):
    return f""" 
        A possible anomaly has been detected in the following location: { data['location'] } at { data['time'] }. 
            Anomaly Audio class: { data['audio_classification'] },
            Anomaly Video class: { data['video_classification'] }
            Image: { data['image_link'] }, 
            Video: { data['video_link'] }. 
        Please take necessary actions if required. 
    """


def notify_owner(
    notification_data={
        "type": "sms",
        "data": {
            "location": "San Francisco",
            "time": "1 PM",
            "audio_classification": "None",
            "video_classification": "None",
            "image_link": "N/A",
            "video_link": "N/A",
        },
    }
):
    try:
        print("notification_data", notification_data)
        client = Client(Config.ACCOUNT_SID, Config.AUTH_TOKEN)
        if notification_data["type"] == "sms":
            send_sms(
                client=client,
                message_template=simple_message_template_b,
                message_data=notification_data["data"],
            )
        elif notification_data["type"] == "call":
            make_call(
                client=client, template_url="http://demo.twilio.com/docs/voice.xml"
            )
        print("Owner notified successfully ✔")
        return True
    except Exception as e:
        print("Unable to send message, possible error:", e)
        # raise e
    # make_call(client)


if __name__ == "__main__":
    notify_owner()

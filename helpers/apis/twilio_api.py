"""
     Twilio API service
"""
from twilio.rest import Client

# import geocoder
import time

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = ""
auth_token = ""

FROM_PHONE_NUMBER = ""
TO_PHONE_NUMBER = ""


def make_call(client, template_url):
    call = client.calls.create(
        url=template_url,
        to=TO_PHONE_NUMBER,
        from_=FROM_PHONE_NUMBER,
    )

    print(call.sid)


def send_sms(client: object, message_template: callable, message_data: dict) -> None:

    message = client.messages.create(
        body=message_template(message_data), from_=FROM_PHONE_NUMBER, to=TO_PHONE_NUMBER
    )

    return message.sid


def simple_message_template_a(data):
    return f""" 
        An anomaly alert! ⚠
        Time: { data['time'] }
        Location: { data['location'] }
        Anomaly Audio classification: { data['audio_classification'] }
        Anomaly Video classification: { data['video_classification'] }
        Check evidence here:
        Image: { data['image_link'] }
        Video: { data['video_link'] }
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


def notify_owner(notification_data: object = {"type": "sms", "data": {""}}):
    try:
        print("notification_data", notification_data)
        client = Client(account_sid, auth_token)
        print("A")
        print(simple_message_template_a(notification_data["data"]))
        print("B")
        if notification_data["type"] == "sms":
            send_sms(
                client=client,
                message_template=simple_message_template_a,
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
